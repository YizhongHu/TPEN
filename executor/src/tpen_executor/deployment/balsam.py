"""Generated Balsam deployment data and explicit application policy.

Descriptor construction and rendering are pure.  Site creation and app sync
are available only through :func:`deploy_owned_site`, which requires an
injected backend and performs identity readback before and after each effect.
The optional SDK is intentionally absent from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any, ClassVar, Protocol

from .._codec import canonical_bytes, decode_json, digest, fields, sha256_text, text
from .._records import frozen_record
from ..environments import PreparedPythonEnvironment
from ..scheduling.pbs import PbsBatchResources, PbsVenue


class BalsamDeploymentRefusal(ValueError):
    """Raised when a deployment is malformed or its identity readback conflicts."""


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise BalsamDeploymentRefusal(f"{name} must be an ASCII identifier-like string")
    return value


def _absolute(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip() or not Path(value).is_absolute():
        raise BalsamDeploymentRefusal(f"{name} must be an absolute path")
    return value


def _inside(path: str, parent: str) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(parent).resolve(strict=False))
    except ValueError:
        return False
    return True


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise BalsamDeploymentRefusal(f"{name} must be a string-keyed mapping")
    return value


@frozen_record
class LockedControlEnvironment:
    """Identity of the separately prepared control interpreter."""

    schema: ClassVar[str] = "tpen-executor/locked-control-environment-v1"
    lock: str
    lock_sha256: str
    prefix: str
    environment: PreparedPythonEnvironment

    def __post_init__(self) -> None:
        object.__setattr__(self, "lock", text(self.lock, "lock"))
        sha256_text(self.lock_sha256, "lock_sha256")
        prefix = _absolute(self.prefix, "prefix")
        if _inside(prefix, str(Path.home())):
            raise BalsamDeploymentRefusal("control prefix must be outside the home directory")
        if not isinstance(self.environment, PreparedPythonEnvironment):
            raise TypeError("environment must be PreparedPythonEnvironment")
        if not _inside(self.environment.python, prefix):
            raise BalsamDeploymentRefusal("control interpreter must be inside control prefix")
        object.__setattr__(self, "prefix", prefix)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""
        return {"schema": self.schema, "lock": self.lock, "lock_sha256": self.lock_sha256,
                "prefix": self.prefix, "environment": self.environment.to_dict()}

    @classmethod
    def from_mapping(cls, value: Any) -> "LockedControlEnvironment":
        """Decode one strict control-environment record."""
        data = fields(value, {"schema", "lock", "lock_sha256", "prefix", "environment"},
                      cls.schema, cls.__name__)
        return cls(data["lock"], data["lock_sha256"], data["prefix"],
                   PreparedPythonEnvironment.from_mapping(data["environment"]))


@frozen_record
class BalsamDeployment:
    """Effect-free deployment identity and owned-site references."""

    schema: ClassVar[str] = "tpen-executor/balsam-deployment-v1"
    venue: PbsVenue
    site_template: str
    api_root: str
    credential_file: str
    site_name: str
    site_directory: str
    control_environment: LockedControlEnvironment

    def __post_init__(self) -> None:
        if not isinstance(self.venue, PbsVenue):
            raise TypeError("venue must be PbsVenue")
        object.__setattr__(self, "site_template", _identifier(self.site_template, "site_template"))
        object.__setattr__(self, "api_root", text(self.api_root, "api_root"))
        object.__setattr__(self, "credential_file", _absolute(self.credential_file, "credential_file"))
        object.__setattr__(self, "site_name", _identifier(self.site_name, "site_name"))
        site_directory = _absolute(self.site_directory, "site_directory")
        if Path(site_directory).name != self.site_name:
            raise BalsamDeploymentRefusal("site_directory must end with the exact site_name")
        if not isinstance(self.control_environment, LockedControlEnvironment):
            raise TypeError("control_environment must be LockedControlEnvironment")
        object.__setattr__(self, "site_directory", site_directory)

    @property
    def deployment_digest(self) -> str:
        """Return the immutable deployment identity digest."""
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation without secret contents."""
        return {"schema": self.schema, "venue": self.venue.to_dict(), "site_template": self.site_template,
                "api_root": self.api_root, "credential_file": self.credential_file,
                "site_name": self.site_name, "site_directory": self.site_directory,
                "control_environment": self.control_environment.to_dict()}

    def encode(self) -> bytes:
        """Encode the canonical deployment description."""
        return canonical_bytes(self.to_dict())

    def render(self, resources: PbsBatchResources | None = None) -> "DeploymentRender":
        """Render this descriptor without performing an external effect."""
        return render_deployment(self, resources)

    def deploy(self, backend: "DeploymentBackend",
               resources: PbsBatchResources | None = None) -> tuple["DeploymentRender", "DeploymentReadback"]:
        """Perform the explicit owned-site deployment through an injected backend."""
        return deploy_owned_site(self, backend, resources)

    @classmethod
    def from_mapping(cls, value: Any) -> "BalsamDeployment":
        """Decode one strict deployment description."""
        data = fields(value, {"schema", "venue", "site_template", "api_root", "credential_file",
                              "site_name", "site_directory", "control_environment"}, cls.schema, cls.__name__)
        venue = data["venue"] if isinstance(data["venue"], PbsVenue) else PbsVenue.from_mapping(data["venue"])
        return cls(venue,
                   data["site_template"], data["api_root"], data["credential_file"],
                   data["site_name"], data["site_directory"],
                   LockedControlEnvironment.from_mapping(data["control_environment"]))

    @classmethod
    def decode(cls, value: bytes | str | Mapping[str, Any]) -> "BalsamDeployment":
        """Decode canonical JSON bytes or a mapping."""
        return cls.from_mapping(decode_json(value))


@frozen_record
class DeploymentRender:
    """Deterministic generated site, app, and carrier template artifacts."""

    schema: ClassVar[str] = "tpen-executor/balsam-deployment-render-v1"
    deployment_digest: str
    application_name: str
    settings: str
    application: str
    job_template: str

    def __post_init__(self) -> None:
        sha256_text(self.deployment_digest, "deployment_digest")
        _identifier(self.application_name, "application_name")
        for name in ("settings", "application", "job_template"):
            text(getattr(self, name), name)

    @property
    def artifact_digests(self) -> Mapping[str, str]:
        """Return digests for each generated artifact."""
        return {name: digest(getattr(self, name)) for name in ("settings", "application", "job_template")}

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical generated artifact record."""
        return {"schema": self.schema, "deployment_digest": self.deployment_digest,
                "application_name": self.application_name, "settings": self.settings,
                "application": self.application, "job_template": self.job_template,
                "artifact_digests": dict(self.artifact_digests)}

    def encode(self) -> bytes:
        """Encode the canonical render record."""
        return canonical_bytes(self.to_dict())


@frozen_record
class DeploymentReadback:
    """Non-secret effective identity returned after explicit deployment."""

    schema: ClassVar[str] = "tpen-executor/balsam-deployment-readback-v1"
    site_name: str
    site_digest: str
    application_name: str
    application_digest: str
    deployment_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_name", _identifier(self.site_name, "site_name"))
        object.__setattr__(self, "application_name", _identifier(self.application_name, "application_name"))
        for name in ("site_digest", "application_digest", "deployment_digest"):
            sha256_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        """Return the non-secret readback record."""
        return {"schema": self.schema, "site_name": self.site_name, "site_digest": self.site_digest,
                "application_name": self.application_name, "application_digest": self.application_digest,
                "deployment_digest": self.deployment_digest}


class DeploymentBackend(Protocol):
    """Finite injected API used by explicit site/app deployment."""

    def read_site(self, site_name: str) -> Mapping[str, Any] | None:
        """Read one exact site identity, or return None when absent."""
        ...

    def create_site(self, site_name: str, template: str, site_directory: str) -> Mapping[str, Any]:
        """Create the exact owned site."""
        ...

    def read_application(self, site_name: str, application_name: str) -> Mapping[str, Any] | None:
        """Read one exact application identity, or return None when absent."""
        ...

    def sync_application(self, site_name: str, application_name: str, source: str) -> Mapping[str, Any]:
        """Synchronize one generated application."""
        ...


def _record_digest(record: Mapping[str, Any], name: str) -> str:
    """Digest a backend record after rejecting secret-shaped fields."""
    data = _mapping(record, name)
    forbidden = {"token", "access_token", "client_secret", "password", "secret"}
    if forbidden & set(data):
        raise BalsamDeploymentRefusal(f"{name} readback contains protected credential data")
    return digest(data)


def _require_identity(record: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    """Require every deployment identity field and reject crossed values."""
    for key, value in expected.items():
        if key not in record:
            raise BalsamDeploymentRefusal(f"{name} readback omits required {key}")
        if record[key] != value:
            raise BalsamDeploymentRefusal(f"{name} readback conflicts on {key}")


_SOURCE_DIGEST_TOKEN = "__TPEN_APPLICATION_SOURCE_DIGEST__"
_SOURCE_DIGEST_LINE = re.compile(r"^SOURCE_DIGEST = ('[0-9a-f]{64}')$", re.MULTILINE)


def _application_source_digest(source: str) -> str:
    """Return the digest of the source before its explicit digest binding."""
    match = _SOURCE_DIGEST_LINE.search(source)
    if match is None:
        raise BalsamDeploymentRefusal("generated application omits its source digest binding")
    canonical = source[:match.start(1)] + repr(_SOURCE_DIGEST_TOKEN) + source[match.end(1):]
    return digest(canonical)


def _render_application(spec: BalsamDeployment, application_name: str) -> str:
    """Render the concrete control-side application policy source."""
    source = (
        '"""Generated application; identity and lifecycle policy are fixed."""\n'
        "from balsam.api import ApplicationDefinition as _PinnedApplicationDefinition\n"
        "from tpen_balsam_control.application import make_application_definition\n"
        "from tpen_executor.lifecycle.balsam import GroupExitContract, group_exit_code\n\n"
        "class _PinnedAPI:\n"
        "    ApplicationDefinition = _PinnedApplicationDefinition\n\n"
        f"SITE_NAME = {spec.site_name!r}\n"
        f"APPLICATION_NAME = {application_name!r}\n"
        f"DEPLOYMENT_DIGEST = {spec.deployment_digest!r}\n"
        f"SOURCE_DIGEST = {_SOURCE_DIGEST_TOKEN!r}\n"
        f"CONTROL_PYTHON = {spec.control_environment.environment.python!r}\n"
        "POLICY_OVERRIDES = {\n"
        "    'backend_retry_codes': (1,),\n"
        "    'no_retry_code': 2,\n"
        "    'timeout_state': 'failed',\n"
        "    'error_state': 'failed',\n"
        "    'rearm': 'refuse',\n"
        "}\n\n"
        "def _invoke_group(job):\n"
        "    callback = getattr(job, 'tpen_run_group_controller', None)\n"
        "    if not callable(callback):\n"
        "        raise RuntimeError('generated application has no bound group controller')\n"
        "    return callback()\n\n"
        "def _record_failure(job, error):\n"
        "    callback = getattr(job, 'tpen_record_failure', None)\n"
        "    if not callable(callback):\n"
        "        raise RuntimeError('generated application has no failure recorder')\n"
        "    callback(error)\n\n"
        "def _record_observation(job, value):\n"
        "    callback = getattr(job, 'tpen_record_observation', None)\n"
        "    if not callable(callback):\n"
        "        raise RuntimeError('generated application has no observation recorder')\n"
        "    callback(value)\n\n"
        "ApplicationDefinition = make_application_definition(\n"
        "    _PinnedAPI,\n"
        "    invoke_group=_invoke_group,\n"
        "    exit_code=group_exit_code,\n"
        "    contract=GroupExitContract(tuple(POLICY_OVERRIDES['backend_retry_codes']),\n"
        "                                POLICY_OVERRIDES['no_retry_code']),\n"
        "    record_failure=_record_failure,\n"
        "    record_observation=_record_observation,\n"
        "    site_name=SITE_NAME,\n"
        "    application_name=APPLICATION_NAME,\n"
        "    deployment_digest=DEPLOYMENT_DIGEST,\n"
        "    source_digest=SOURCE_DIGEST,\n"
        "    policy_overrides=POLICY_OVERRIDES,\n"
        ")\n"
        "\n"
    )
    source_digest = digest(source)
    return source.replace(repr(_SOURCE_DIGEST_TOKEN), repr(source_digest), 1)


def _render_job_template(spec: BalsamDeployment) -> str:
    """Render the bounded carrier template without a submit command."""
    filesystems = ":".join(spec.venue.filesystems)
    return (
        "# generated carrier template; rendering has no scheduler effect\n"
        f"select={{num_nodes}}:system={spec.venue.system}:ncpus={{ncpus}}:ngpus={{ngpus}}\n"
        "walltime={{walltime_minutes}}\n"
        "account={{account}}\n"
        "queue={{queue}}\n"
        f"filesystems={filesystems}\n"
        "rerunnable=false\n"
        "stdout={{tpen_stdout}}\n"
        "stderr={{tpen_stderr}}\n"
    )


def render_deployment(spec: BalsamDeployment, resources: PbsBatchResources | None = None) -> DeploymentRender:
    """Render deterministic artifacts without login, site, app, or submit effects."""
    if not isinstance(spec, BalsamDeployment):
        raise TypeError("spec must be BalsamDeployment")
    if resources is not None:
        resources.validate(spec.venue)
    application_name = f"tpen_group_{spec.deployment_digest[:12]}"
    settings = json.dumps({
        "schema": "tpen-executor/balsam-site-settings-v1",
        "site_name": spec.site_name,
        "site_directory": spec.site_directory,
        "site_template": spec.site_template,
        "api_root": spec.api_root,
        "credential_file": spec.credential_file,
        "control_python": spec.control_environment.environment.python,
        "deployment_digest": spec.deployment_digest,
    }, sort_keys=True, separators=(",", ":"))
    return DeploymentRender(spec.deployment_digest, application_name, settings,
                            _render_application(spec, application_name), _render_job_template(spec))


def deploy_owned_site(spec: BalsamDeployment, backend: DeploymentBackend,
                      resources: PbsBatchResources | None = None) -> tuple[DeploymentRender, DeploymentReadback]:
    """Explicitly create/read back one owned site and one generated app.

    Existing identities are preserved.  A mismatched active site or app is a
    refusal; this function never overwrites or silently switches identities.
    """
    if not isinstance(spec, BalsamDeployment):
        raise TypeError("spec must be BalsamDeployment")
    if not all(callable(getattr(backend, name, None)) for name in
               ("read_site", "create_site", "read_application", "sync_application")):
        raise TypeError("backend does not implement the finite deployment API")
    rendered = render_deployment(spec, resources)
    source_digest = _application_source_digest(rendered.application)
    expected_identity = {
        "site_name": spec.site_name,
        "site_directory": spec.site_directory,
        "site_template": spec.site_template,
        "source_digest": source_digest,
        "deployment_digest": spec.deployment_digest,
    }
    site = backend.read_site(spec.site_name)
    if site is None:
        backend.create_site(spec.site_name, spec.site_template, spec.site_directory)
        site = backend.read_site(spec.site_name)
    if site is None:
        raise BalsamDeploymentRefusal("site readback is absent after creation")
    site = _mapping(site, "site readback")
    _require_identity(site, expected_identity, "site")
    site_digest = _record_digest(site, "site readback")
    app = backend.read_application(spec.site_name, rendered.application_name)
    if app is None:
        backend.sync_application(spec.site_name, rendered.application_name, rendered.application)
        app = backend.read_application(spec.site_name, rendered.application_name)
    if app is None:
        raise BalsamDeploymentRefusal("application readback is absent after synchronization")
    app = _mapping(app, "application readback")
    _require_identity(app, {**expected_identity, "application_name": rendered.application_name}, "application")
    app_digest = _record_digest(app, "application readback")
    return rendered, DeploymentReadback(spec.site_name, site_digest, rendered.application_name,
                                        app_digest, spec.deployment_digest)


def doctor(spec: BalsamDeployment) -> dict[str, str]:
    """Report non-secret expected deployment metadata without reauthentication."""
    if not isinstance(spec, BalsamDeployment):
        raise TypeError("spec must be BalsamDeployment")
    return {"api_root": spec.api_root, "credential_file": spec.credential_file,
            "site_name": spec.site_name, "site_template": spec.site_template,
            "deployment_digest": spec.deployment_digest,
            "credential_policy": "external-file-not-read"}


__all__ = [
    "BalsamDeployment", "BalsamDeploymentRefusal", "DeploymentBackend", "DeploymentReadback",
    "DeploymentRender", "LockedControlEnvironment", "deploy_owned_site", "doctor", "render_deployment",
]
