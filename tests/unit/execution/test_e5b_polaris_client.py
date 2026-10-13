"""Source tests for the lazy real-Balsam E5b deployment wrapper."""

from __future__ import annotations

import subprocess
import sys
import types
import importlib
import json
from pathlib import Path

from tpen.execution import BalsamJobState
from tools.e5b_polaris_client import PolarisBalsamClient


class FakeApp:
    """Small ApplicationDefinition-shaped fake used without Balsam installed."""

    site = None
    environment_variables = {}
    client = None
    sync_calls = 0
    submitted = []

    @classmethod
    def _set_client(cls, client):
        cls.client = client

    @classmethod
    def sync(cls):
        cls.sync_calls += 1

    @classmethod
    def submit(cls, **kwargs):
        cls.submitted.append(kwargs)
        return types.SimpleNamespace(id=730099)


class FakeJobManager:
    def __init__(self):
        self.last_id = None
        self.job = types.SimpleNamespace(
            id=730099,
            state="restart_ready",
            state_data={"run_count": 2},
            batch_job_id=991,
            return_code=70,
            workdir=Path("e5b-polaris-job"),
        )

    def get(self, **kwargs):
        self.last_id = kwargs["id"]
        return self.job


def _fake_modules(monkeypatch):
    manager = FakeJobManager()
    api = types.ModuleType("balsam.api")
    api.Job = types.SimpleNamespace(objects=manager)
    config = types.ModuleType("balsam.config")

    class SiteConfig:
        def __init__(self, site_path):
            self.site_path = Path(site_path)
            self.site_id = 784
            self.client = object()

    config.SiteConfig = SiteConfig
    balsam = types.ModuleType("balsam")
    balsam.api = api
    monkeypatch.setitem(sys.modules, "balsam", balsam)
    monkeypatch.setitem(sys.modules, "balsam.api", api)
    monkeypatch.setitem(sys.modules, "balsam.config", config)
    app_module = types.ModuleType("fake_e5b_app")
    app_module.PolarisE5bApp = FakeApp
    monkeypatch.setitem(sys.modules, "fake_e5b_app", app_module)
    return manager


def _client(tmp_path: Path) -> PolarisBalsamClient:
    return PolarisBalsamClient(
        tmp_path / "site",
        {
            "app_module": "fake_e5b_app",
            "app_class": "PolarisE5bApp",
            "checkout_root": "/checkout",
            "site_packages": "/site-packages",
            "pilot_root": "/pilot",
            "job_data": {"pilot_root": "/pilot"},
            "workdir": "e5b-job",
            "wall_time_min": 5,
        },
    )


def test_wrapper_is_lazy_and_integer_ack_is_canonicalized(monkeypatch, tmp_path):
    _fake_modules(monkeypatch)
    client = _client(tmp_path)
    ack = client.register_job(lineage_id="lineage", batch=object(), backend_options=client.backend_options)
    assert ack == {"id": 730099}
    assert FakeApp.site == 784
    assert FakeApp.sync_calls == 1
    assert FakeApp.submitted[-1]["data"]["pilot_root"] == "/pilot"


def test_observation_preserves_backend_state_and_integer_lookup(monkeypatch, tmp_path):
    manager = _fake_modules(monkeypatch)
    result = _client(tmp_path).observe_job("730099")
    assert manager.last_id == 730099
    assert result["job_id"] == 730099
    assert result["state"] == BalsamJobState.ALLOCATION_LOST.value
    assert result["metadata"]["run_count"] == 2


def test_importing_wrapper_does_not_import_optional_balsam():
    code = "import sys; import tools.e5b_polaris_client; assert 'balsam' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_worker_environment_prepends_without_discarding_facility_paths(monkeypatch, tmp_path):
    """The private additions must retain the inherited MPI environment."""

    monkeypatch.setenv("LD_LIBRARY_PATH", "/facility/libfabric")
    monkeypatch.setenv("PATH", "/facility/bin")
    monkeypatch.setenv("PYTHONPATH", "/facility/python")
    client = _client(tmp_path)
    environment = client._worker_environment()
    assert environment["LD_LIBRARY_PATH"].endswith(":/facility/libfabric")
    assert environment["PATH"].endswith(":/facility/bin")
    assert environment["PYTHONPATH"].endswith(":/facility/python")


def _load_fake_app(monkeypatch):
    """Load the qualification app against a fake Balsam module."""

    class FakeApplicationDefinition:
        def __init__(self, job):
            self.job = job

    api = types.ModuleType("balsam.api")
    api.ApplicationDefinition = FakeApplicationDefinition
    balsam = types.ModuleType("balsam")
    balsam.api = api
    monkeypatch.setitem(sys.modules, "balsam", balsam)
    monkeypatch.setitem(sys.modules, "balsam.api", api)
    sys.modules.pop("tools.e5b_polaris_app", None)
    return importlib.import_module("tools.e5b_polaris_app")


def test_app_records_pre_e3_failure_and_uses_non_retry_exit(monkeypatch, tmp_path):
    """Bootstrap failure is durable, non-success, and outside Balsam retry codes."""

    app_module = _load_fake_app(monkeypatch)
    failing_runner = types.ModuleType("tools.e5b_polaris_runner")

    def fail(**_kwargs):
        raise ImportError("synthetic pre-E3 import failure")

    failing_runner.run_attempt = fail
    monkeypatch.setitem(sys.modules, "tools.e5b_polaris_runner", failing_runner)
    app = app_module.PolarisE5bApp(
        types.SimpleNamespace(id=42609999, data={"pilot_root": str(tmp_path)})
    )
    try:
        app.run()
    except SystemExit as exc:
        assert exc.code == app_module.PRE_E3_FAILURE_EXIT_CODE == 2
    else:  # pragma: no cover - assertion documents the required process status
        raise AssertionError("pre-E3 failure returned successfully")
    record = json.loads((tmp_path / "pre-e3-failure.json").read_text())
    assert record["exception_type"] == "ImportError"
    assert record["reason"].startswith("Balsam application failed before")


def test_app_converts_park_result_to_non_retry_exit(monkeypatch, tmp_path):
    """A worker park must not become Balsam's successful zero exit."""

    app_module = _load_fake_app(monkeypatch)
    parked_runner = types.ModuleType("tools.e5b_polaris_runner")
    parked_runner.run_attempt = lambda **_kwargs: 3
    monkeypatch.setitem(sys.modules, "tools.e5b_polaris_runner", parked_runner)
    app = app_module.PolarisE5bApp(
        types.SimpleNamespace(id=42610000, data={"pilot_root": str(tmp_path)})
    )
    try:
        app.run()
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("park result returned successfully")
    assert not (tmp_path / "pre-e3-failure.json").exists()
