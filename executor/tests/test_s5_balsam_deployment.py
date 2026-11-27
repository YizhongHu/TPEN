"""Focused S5 deployment identity and application-policy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.deployment.balsam import (
    BalsamDeployment,
    BalsamDeploymentRefusal,
    LockedControlEnvironment,
    deploy_owned_site,
    doctor,
    render_deployment,
)
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.lifecycle.balsam import (
    BalsamGroupRefusal,
    GroupExitContract,
    make_group_application,
)
from tpen_executor.scheduling.pbs import PackedNodeLayout, PbsBatchResources, PbsQueueLimit, PbsVenue

CONTROL_SOURCE = Path(__file__).parents[1] / "control" / "balsam" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from tpen_balsam_control.application import make_application_definition  # noqa: E402


def _spec(tmp_path: Path) -> BalsamDeployment:
    venue = PbsVenue(
        "machine",
        ("shared",),
        (PbsQueueLimit("short", 2, 60, 3),),
        PackedNodeLayout(4, 8, 2, ("0-1", "2-3")),
    )
    prefix = tmp_path / "control-prefix"
    return BalsamDeployment(
        venue,
        "template",
        "https://service.invalid",
        str(tmp_path / "client.yml"),
        "owned-site",
        str(tmp_path / "sites" / "owned-site"),
        LockedControlEnvironment(
            "control/uv.lock",
            digest("lock"),
            str(prefix),
            PreparedPythonEnvironment(str(prefix / "bin" / "python"), SourceScript(str(tmp_path / "setup.sh"))),
        ),
    )


class _Backend:
    def __init__(self) -> None:
        self.sites: dict[str, dict[str, str]] = {}
        self.apps: dict[tuple[str, str], dict[str, str]] = {}
        self.calls: list[str] = []

    def read_site(self, site_name: str):
        return self.sites.get(site_name)

    def create_site(self, site_name: str, template: str, site_directory: str):
        self.calls.append("create-site")
        value = {"name": site_name, "template": template, "directory": site_directory}
        self.sites[site_name] = value
        return value

    def read_application(self, site_name: str, application_name: str):
        return self.apps.get((site_name, application_name))

    def sync_application(self, site_name: str, application_name: str, source: str):
        self.calls.append("sync-application")
        value = {"name": application_name, "source_digest": digest(source)}
        self.apps[(site_name, application_name)] = value
        return value


def test_render_is_deterministic_and_effect_free(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    resources = PbsBatchResources("account", "short", 1, 15)
    first = render_deployment(spec, resources)
    second = render_deployment(spec, resources)
    assert first == second
    assert "token" not in first.settings.lower()
    assert "handle_timeout" in first.application
    assert "handle_error" in first.application


def test_deploy_preserves_existing_identity_and_returns_effective_digests(tmp_path: Path) -> None:
    backend = _Backend()
    spec = _spec(tmp_path)
    rendered, readback = deploy_owned_site(spec, backend, PbsBatchResources("account", "short", 1, 15))
    assert backend.calls == ["create-site", "sync-application"]
    assert readback.deployment_digest == spec.deployment_digest
    deploy_owned_site(spec, backend, PbsBatchResources("account", "short", 1, 15))
    assert backend.calls == ["create-site", "sync-application"]
    assert rendered.artifact_digests["application"] == digest(rendered.application)


def test_deploy_refuses_crossed_site_and_credential_readback(tmp_path: Path) -> None:
    backend = _Backend()
    spec = _spec(tmp_path)
    backend.sites[spec.site_name] = {"name": "different-site"}
    with pytest.raises(BalsamDeploymentRefusal):
        deploy_owned_site(spec, backend)
    backend.sites[spec.site_name] = {"name": spec.site_name, "token": "secret"}
    with pytest.raises(BalsamDeploymentRefusal):
        deploy_owned_site(spec, backend)
    assert doctor(spec)["credential_policy"] == "external-file-not-read"


def test_control_environment_refuses_home_prefix(tmp_path: Path) -> None:
    with pytest.raises(BalsamDeploymentRefusal):
        LockedControlEnvironment(
            "control/uv.lock", digest("lock"), str(Path.home() / "prefix"),
            PreparedPythonEnvironment(str(Path.home() / "prefix" / "bin" / "python")),
        )


def test_scheduler_records_refuse_overlapping_slots_and_queue_overflow() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        PackedNodeLayout(4, 8, 2, ("0-2", "2-3"))
    venue = PbsVenue("machine", ("shared",), (PbsQueueLimit("short", 1, 10, 1),),
                     PackedNodeLayout(4, 8, 2, ("0-1", "2-3")))
    with pytest.raises(ValueError, match="walltime"):
        PbsBatchResources("account", "short", 1, 11).validate(venue)


def test_application_hooks_mark_failure_and_refuse_rearm() -> None:
    events: list[tuple[str, object]] = []
    app_type = make_group_application(
        SimpleNamespace(ApplicationDefinition=object),
        controller=lambda job: None,
        contract=GroupExitContract((1,)),
        record_failure=lambda job, error: events.append(("failure", error)),
        timeout_handler=lambda job: events.append(("timeout", job)),
        error_handler=lambda job, error: events.append(("error", error)),
    )
    app = app_type()
    app.job = SimpleNamespace(state="running")
    app.handle_timeout()
    assert app.job.state == "failed"
    app.handle_error(RuntimeError("bad"))
    assert app.job.state == "failed"
    with pytest.raises(BalsamGroupRefusal):
        app.request_rearm()
    assert [event[0] for event in events] == ["timeout", "error"]


def test_control_application_hooks_are_explicit() -> None:
    observations: list[object] = []
    application_type = make_application_definition(
        SimpleNamespace(ApplicationDefinition=object),
        invoke_group=lambda job: object(),
        exit_code=lambda outcome, contract: 0,
        contract=object(),
        record_failure=lambda job, error: None,
        record_observation=lambda job, value: observations.append(value),
        application_name="generated-app",
        deployment_digest="a" * 64,
    )
    app = application_type()
    app.job = SimpleNamespace(state="running")
    app.handle_timeout()
    app.handle_error()
    assert app.job.state == "failed"
    assert observations == ["timeout-awaiting-explicit-reentry", "error-awaiting-explicit-reentry"]
    with pytest.raises(Exception, match="grant-bound"):
        app.request_rearm()

