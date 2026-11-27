"""Focused S5 deployment identity and application-policy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

from tpen_executor._codec import digest
from tpen_executor.deployment.balsam import (
    BalsamDeployment,
    BalsamDeploymentRefusal,
    LockedControlEnvironment,
    _application_source_digest,
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
    def __init__(self, spec: BalsamDeployment, rendered) -> None:
        self.sites: dict[str, dict[str, str]] = {}
        self.apps: dict[tuple[str, str], dict[str, str]] = {}
        self.calls: list[str] = []
        self.reads: list[str] = []
        self.site_after_create = None
        self.app_after_sync = None
        self.site_record = {
            "site_name": spec.site_name,
            "site_directory": spec.site_directory,
            "site_template": spec.site_template,
            "source_digest": _application_source_digest(rendered.application),
            "deployment_digest": spec.deployment_digest,
        }
        self.app_record = {
            **self.site_record,
            "application_name": rendered.application_name,
        }

    def read_site(self, site_name: str):
        self.reads.append("read-site")
        return self.sites.get(site_name)

    def create_site(self, site_name: str, template: str, site_directory: str):
        self.calls.append("create-site")
        value = dict(self.site_after_create or self.site_record)
        self.sites[site_name] = value
        return value

    def read_application(self, site_name: str, application_name: str):
        self.reads.append("read-application")
        return self.apps.get((site_name, application_name))

    def sync_application(self, site_name: str, application_name: str, source: str):
        self.calls.append("sync-application")
        value = dict(self.app_after_sync or self.app_record)
        self.apps[(site_name, application_name)] = value
        return value


def test_render_is_deterministic_and_effect_free(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    resources = PbsBatchResources("account", "short", 1, 15)
    first = render_deployment(spec, resources)
    second = render_deployment(spec, resources)
    assert first == second
    assert "token" not in first.settings.lower()
    assert "make_application_definition(" in first.application
    assert "POLICY_OVERRIDES" in first.application


def test_deploy_preserves_existing_identity_and_returns_effective_digests(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    rendered, readback = deploy_owned_site(spec, backend, PbsBatchResources("account", "short", 1, 15))
    assert backend.calls == ["create-site", "sync-application"]
    assert backend.reads == ["read-site", "read-site", "read-application", "read-application"]
    assert readback.deployment_digest == spec.deployment_digest
    deploy_owned_site(spec, backend, PbsBatchResources("account", "short", 1, 15))
    assert backend.calls == ["create-site", "sync-application"]
    assert rendered.artifact_digests["application"] == digest(rendered.application)


def test_deploy_refuses_crossed_site_and_credential_readback(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    backend.sites[spec.site_name] = {**backend.site_record, "site_name": "different-site"}
    with pytest.raises(BalsamDeploymentRefusal):
        deploy_owned_site(spec, backend)
    backend.sites[spec.site_name] = {**backend.site_record, "token": "secret"}
    with pytest.raises(BalsamDeploymentRefusal):
        deploy_owned_site(spec, backend)
    assert doctor(spec)["credential_policy"] == "external-file-not-read"


def test_deploy_refuses_crossed_application_site_name(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    # Both records exist already: the crossed application must be refused
    # before deployment considers any mutating operation.
    backend.sites[spec.site_name] = dict(backend.site_record)
    backend.apps[(spec.site_name, rendered.application_name)] = {
        **backend.app_record,
        "site_name": "foreign-site",
    }
    with pytest.raises(BalsamDeploymentRefusal):
        deploy_owned_site(spec, backend)
    assert backend.calls == []


@pytest.mark.parametrize("missing", ["site_directory", "site_template", "source_digest", "deployment_digest"])
def test_deploy_requires_complete_site_identity(tmp_path: Path, missing: str) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    backend.sites[spec.site_name] = dict(backend.site_record)
    backend.sites[spec.site_name].pop(missing)
    with pytest.raises(BalsamDeploymentRefusal, match=missing):
        deploy_owned_site(spec, backend)


@pytest.mark.parametrize("missing", ["site_name", "site_directory", "site_template", "source_digest", "deployment_digest"])
def test_deploy_requires_complete_application_identity(tmp_path: Path, missing: str) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    backend.sites[spec.site_name] = dict(backend.site_record)
    backend.apps[(spec.site_name, rendered.application_name)] = dict(backend.app_record)
    backend.apps[(spec.site_name, rendered.application_name)].pop(missing)
    with pytest.raises(BalsamDeploymentRefusal, match=missing):
        deploy_owned_site(spec, backend)


@pytest.mark.parametrize("field", ["source_digest", "deployment_digest"])
def test_deploy_refuses_changed_application_identity(tmp_path: Path, field: str) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    backend.apps[(spec.site_name, rendered.application_name)] = {
        **backend.app_record,
        field: "0" * 64,
    }
    with pytest.raises(BalsamDeploymentRefusal, match=field):
        deploy_owned_site(spec, backend)


@pytest.mark.parametrize("kind", ["site", "application"])
def test_deploy_requires_post_mutation_readback(tmp_path: Path, kind: str) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    backend = _Backend(spec, rendered)
    if kind == "site":
        def create_without_readback(site_name, template, site_directory):
            backend.calls.append("create-site")
            return dict(backend.site_record)

        backend.create_site = create_without_readback
        # Remove the effective record after the acknowledgement, simulating a
        # backend that accepted the mutation but exposes no readback state.
        def read_site_absent(site_name):
            backend.reads.append("read-site")
            return None

        backend.read_site = read_site_absent
        # Keep an existing valid application so a mutant that trusts the
        # create acknowledgement cannot fail later for an unrelated reason.
        backend.apps[(spec.site_name, rendered.application_name)] = dict(backend.app_record)
        with pytest.raises(BalsamDeploymentRefusal, match="site readback"):
            deploy_owned_site(spec, backend)
        assert backend.calls == ["create-site"]
    else:
        backend.sites[spec.site_name] = dict(backend.site_record)
        backend.app_after_sync = dict(backend.app_record)

        def read_application_conflict(site_name, application_name):
            backend.reads.append("read-application")
            return {**backend.app_record, "deployment_digest": "0" * 64}

        backend.read_application = read_application_conflict
        with pytest.raises(BalsamDeploymentRefusal, match="deployment_digest"):
            deploy_owned_site(spec, backend)
        assert backend.calls == ["sync-application"]


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
        site_name="owned-site",
        application_name="generated-app",
        deployment_digest="a" * 64,
        source_digest="b" * 64,
        policy_overrides={"rearm": "refuse"},
    )
    app = application_type()
    app.job = SimpleNamespace(state="running")
    app.handle_timeout()
    app.handle_error()
    assert app.job.state == "failed"
    assert observations == ["timeout-awaiting-explicit-reentry", "error-awaiting-explicit-reentry"]
    with pytest.raises(Exception, match="grant-bound"):
        app.request_rearm()

def test_rendered_application_is_a_concrete_pinned_consumer(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    rendered = render_deployment(spec)
    module_name = "_tpen_rendered_balsam_app"
    fake_balsam = types.ModuleType("balsam")
    fake_api = types.ModuleType("balsam.api")

    class PinnedApplicationDefinition:
        def run(self):
            raise AssertionError("inherited run hook reached")

        def handle_timeout(self):
            raise AssertionError("inherited timeout hook reached")

        def handle_error(self, error=None):
            raise AssertionError("inherited error hook reached")

    fake_api.ApplicationDefinition = PinnedApplicationDefinition
    fake_balsam.api = fake_api
    old_modules = {name: sys.modules.get(name) for name in ("balsam", "balsam.api")}
    sys.modules["balsam"] = fake_balsam
    sys.modules["balsam.api"] = fake_api
    namespace = {"__name__": module_name}
    try:
        exec(compile(rendered.application, "<rendered-balsam-application>", "exec"), namespace)
    finally:
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    application = namespace["ApplicationDefinition"]
    assert issubclass(application, PinnedApplicationDefinition)
    assert application.tpen_site_name == spec.site_name
    assert application.tpen_application_name == rendered.application_name
    assert application.tpen_deployment_digest == spec.deployment_digest
    assert application.tpen_source_digest == _application_source_digest(rendered.application)
    observations: list[object] = []
    job = SimpleNamespace(
        state="running",
        tpen_run_group_controller=lambda: __import__("tpen_executor.lifecycle.balsam", fromlist=["GroupFailure"]).GroupFailure.PRE_START,
        tpen_record_failure=lambda error: None,
        tpen_record_observation=observations.append,
    )
    app = application()
    app.job = job
    with pytest.raises(SystemExit) as raised:
        app.run()
    assert raised.value.code == 2
    app.handle_timeout()
    app.handle_error(RuntimeError("bad"))
    assert job.state == "failed"
    assert observations[0] == "timeout-awaiting-explicit-reentry"
    assert isinstance(observations[1], RuntimeError)
    with pytest.raises(Exception, match="grant-bound"):
        app.request_rearm()

