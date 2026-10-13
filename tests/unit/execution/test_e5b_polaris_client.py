"""Source tests for the lazy real-Balsam E5b deployment wrapper."""

from __future__ import annotations

import subprocess
import sys
import types
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
