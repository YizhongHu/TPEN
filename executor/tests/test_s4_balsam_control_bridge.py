"""S4 identity, prefix, and finite bridge controls."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import BackendJobId, BatchJobId, GroupId
from tpen_executor.lifecycle.balsam_wire import BalsamWireRequest

CONTROL_SOURCE = Path(__file__).parents[1] / "control" / "balsam" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from tpen_balsam_control.application import (  # noqa: E402
    BalsamControlAmbiguous,
    BalsamControlRefusal,
    FixedBatchOutputPaths,
    SDKIdentity,
    fixed_batch_output_paths,
)
from tpen_balsam_control.bootstrap import build_prefix, prefix_manifest  # noqa: E402
from tpen_balsam_control.bridge import dispatch, load_sdk_operations  # noqa: E402
from tpen_balsam_control.sdk_adapter import require_pinned_version  # noqa: E402


def _sha(seed: str) -> str:
    """Return a valid fixture digest."""

    return digest({"seed": seed})


class FakeOperations:
    """Finite fake SDK adapter used without importing the optional SDK."""

    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.error = error

    def _call(self, operation: str, *values: str) -> str:
        self.calls.append((operation, values))
        if self.error is not None:
            raise self.error
        return {"register": "job", "observe": "queued", "submit": "batch", "rearm": "ready", "readback": "queued"}[operation]

    def register(self, group_id: str, group_digest: str, deployment_digest: str) -> str:
        return self._call("register", group_id, group_digest, deployment_digest)

    def observe(self, group_id: str, backend_job_id: str) -> str:
        return self._call("observe", group_id, backend_job_id)

    def submit(self, group_id: str, backend_job_id: str, batch_digest: str) -> str:
        return self._call("submit", group_id, backend_job_id, batch_digest)

    def rearm(self, group_id: str, backend_job_id: str, recovery_digest: str) -> str:
        return self._call("rearm", group_id, backend_job_id, recovery_digest)

    def readback(self, group_id: str, backend_job_id: str) -> str:
        return self._call("readback", group_id, backend_job_id)


@pytest.mark.parametrize("operation", ["register", "observe", "submit", "rearm", "readback"])
def test_dispatch_maps_only_the_five_closed_operations(operation: str) -> None:
    """Each wire operation reaches its matching injected SDK method."""

    group = GroupId("group")
    job = BackendJobId("job")
    request = {
        "register": BalsamWireRequest.register(group, _sha("group"), _sha("deployment")),
        "observe": BalsamWireRequest.observe(group, job),
        "submit": BalsamWireRequest.submit(group, job, _sha("batch")),
        "rearm": BalsamWireRequest.rearm(group, job, _sha("recovery")),
        "readback": BalsamWireRequest.readback(group, job),
    }[operation]
    fake = FakeOperations()
    result = dispatch(request, fake)
    assert result.status == "ack"
    assert fake.calls[0][0] == operation
    if operation == "register":
        assert result.backend_job_id == BackendJobId("job")
    elif operation == "submit":
        assert result.batch_job_id == BatchJobId("batch")


def test_dispatch_preserves_typed_refusal_and_ambiguous_effect() -> None:
    """Injected failures cannot become successful wire acknowledgements."""

    request = BalsamWireRequest.rearm(GroupId("group"), BackendJobId("job"), _sha("recovery"))
    refused = dispatch(request, FakeOperations(error=BalsamControlRefusal("not admitted")))
    ambiguous = dispatch(request, FakeOperations(error=BalsamControlAmbiguous("lost response")))
    assert refused.status == "refused" and refused.reason == "not admitted"
    assert ambiguous.status == "ambiguous" and ambiguous.reason == "lost response"


def test_fixed_batch_output_paths_reject_arbitrary_optional_parameters() -> None:
    """Only the two generated output destinations are exposed."""

    paths = fixed_batch_output_paths({"tpen_stdout": "/run/out", "tpen_stderr": "/run/err"})
    assert isinstance(paths, FixedBatchOutputPaths)
    assert paths.as_optional_params() == {"tpen_stdout": "/run/out", "tpen_stderr": "/run/err"}
    with pytest.raises(BalsamControlRefusal):
        fixed_batch_output_paths({"tpen_stdout": "/run/out", "tpen_stderr": "/run/err", "extra": "ignored"})
    with pytest.raises(BalsamControlRefusal):
        fixed_batch_output_paths({"tpen_stdout": "relative", "tpen_stderr": "/run/err"})


def test_sdk_identity_binds_every_artifact_digest() -> None:
    """Version and source/lock/artifact identities are mandatory."""

    identity = SDKIdentity("0.7.0a24", "balsam.api", _sha("executor"), _sha("project"), _sha("lock"))
    assert identity.sdk_version == "0.7.0a24"
    with pytest.raises(ValueError):
        SDKIdentity("0.7.0a24", "balsam.api", "not-a-digest", _sha("project"), _sha("lock"))
    assert require_pinned_version("0.7.0.a24") == "0.7.0.a24"
    with pytest.raises(BalsamControlRefusal):
        require_pinned_version("0.7.0a24")


def test_prefix_manifest_is_outside_home_and_refuses_drift(tmp_path: Path) -> None:
    """A final prefix is content-bound and cannot silently change identity."""

    project = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    artifact = tmp_path / "executor.whl"
    python = tmp_path / "python"
    uv = tmp_path / "uv"
    for path, contents in ((project, b"project"), (lock, b"lock"), (artifact, b"artifact"), (python, b"python"), (uv, b"uv")):
        path.write_bytes(contents)
        path.chmod(0o755)
    prefix = tmp_path / "prefix"
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return __import__("subprocess").CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    built = build_prefix(project, lock, artifact, python=python, uv=uv, prefix=prefix, runner=run)
    assert built["prefix"] == str(prefix)
    assert calls[0][0][:2] == (str(uv), "sync")
    assert calls[0][0][2] == "--locked"
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(prefix)
    assert build_prefix(project, lock, artifact, python=python, uv=uv, prefix=prefix, runner=run) == built
    lock.write_bytes(b"changed")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(project, lock, artifact, python=python, uv=uv, prefix=prefix, runner=run)


def test_prefix_manifest_rejects_home_and_unmanifested_prefix(tmp_path: Path) -> None:
    """Home fallback and adoption of an unknown prefix are both refused."""

    project = tmp_path / "project"
    lock = tmp_path / "lock"
    artifact = tmp_path / "artifact"
    python = tmp_path / "python"
    uv = tmp_path / "uv"
    for path in (project, lock, artifact, python, uv):
        path.write_bytes(b"identity")
        path.chmod(0o755)
    with pytest.raises(BalsamControlRefusal, match="outside the home"):
        prefix_manifest(project, lock, artifact, python=python, uv=uv, prefix=Path.home() / "control-prefix")
    prefix = tmp_path / "unmanifested"
    prefix.mkdir()
    (prefix / "foreign").write_bytes(b"foreign")
    with pytest.raises(BalsamControlRefusal, match="unmanifested"):
        build_prefix(project, lock, artifact, python=python, uv=uv, prefix=prefix)


def test_bridge_validates_client_path_before_sdk_import(tmp_path: Path) -> None:
    """Missing client paths fail before optional SDK loading is attempted."""

    with pytest.raises(BalsamControlRefusal):
        load_sdk_operations(str(tmp_path / "missing-client"))
