"""S4 identity, prefix, and finite bridge controls."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import BackendJobId, BatchJobId, GroupId
from tpen_executor.lifecycle.balsam_wire import BalsamWireRequest, decode_result, encode_result

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


def _bootstrap_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create a repository-shaped control project and qualified tools."""

    project = tmp_path / "repo" / "executor" / "control" / "balsam"
    executor_source = project.parents[1] / "src"
    control_source = project / "src"
    executor_source.mkdir(parents=True)
    control_source.mkdir(parents=True)
    (project / "pyproject.toml").write_bytes(b"project")
    (project / "uv.lock").write_bytes(b"lock")
    (executor_source / "executor.py").write_bytes(b"executor")
    (control_source / "bridge.py").write_bytes(b"bridge")
    (project.parents[1] / "pyproject.toml").write_bytes(b"executor project")
    (project.parents[1] / "uv.lock").write_bytes(b"executor lock")
    python = tmp_path / "python"
    uv = tmp_path / "uv"
    for path, contents in ((python, b"python"), (uv, b"uv")):
        path.write_bytes(contents)
        path.chmod(0o755)
    return {
        "project_dir": project,
        "lock": project / "uv.lock",
        "python": python,
        "uv": uv,
        "cache_dir": tmp_path / "uv-cache",
        "prefix": tmp_path / "prefix",
    }


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


@pytest.mark.parametrize("operation", ["register", "submit", "rearm"])
@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_mutating_post_dispatch_errors_are_ambiguous(operation: str, error_type: type[Exception]) -> None:
    """An effect recorded before response parsing must be reconciled."""

    request = {
        "register": BalsamWireRequest.register(GroupId("group"), _sha("group"), _sha("deployment")),
        "submit": BalsamWireRequest.submit(GroupId("group"), BackendJobId("job"), _sha("batch")),
        "rearm": BalsamWireRequest.rearm(GroupId("group"), BackendJobId("job"), _sha("recovery")),
    }[operation]
    fake = FakeOperations(error=error_type("response parsing failed"))
    result = dispatch(request, fake)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == operation
    assert result.status == "ambiguous"
    assert decode_result(encode_result(result)) == result


@pytest.mark.parametrize("operation", ["register", "submit", "rearm"])
def test_malformed_mutating_responses_are_ambiguous(operation: str) -> None:
    """Malformed handles or states after dispatch are unknown effects."""

    class MalformedOperations(FakeOperations):
        def _call(self, operation: str, *values: str) -> str:
            self.calls.append((operation, values))
            return ""

    request = {
        "register": BalsamWireRequest.register(GroupId("group"), _sha("group"), _sha("deployment")),
        "submit": BalsamWireRequest.submit(GroupId("group"), BackendJobId("job"), _sha("batch")),
        "rearm": BalsamWireRequest.rearm(GroupId("group"), BackendJobId("job"), _sha("recovery")),
    }[operation]
    fake = MalformedOperations()
    result = dispatch(request, fake)
    assert fake.calls[0][0] == operation
    assert result.status == "ambiguous"
    assert decode_result(encode_result(result)) == result


@pytest.mark.parametrize(
    "error",
    [
        BalsamControlRefusal("rejected"),
        BalsamControlAmbiguous("uncertain"),
        RuntimeError("sdk failed"),
        TypeError("bad response"),
        ValueError("bad value"),
    ],
    ids=["refusal", "ambiguous", "generic", "type", "value"],
)
@pytest.mark.parametrize("operation", ["register", "observe", "submit", "rearm", "readback"])
def test_all_operation_failures_round_trip(error: Exception, operation: str) -> None:
    """Failure classification is operation-aware and codec-valid."""

    request = {
        "register": BalsamWireRequest.register(GroupId("group"), _sha("group"), _sha("deployment")),
        "observe": BalsamWireRequest.observe(GroupId("group"), BackendJobId("job")),
        "submit": BalsamWireRequest.submit(GroupId("group"), BackendJobId("job"), _sha("batch")),
        "rearm": BalsamWireRequest.rearm(GroupId("group"), BackendJobId("job"), _sha("recovery")),
        "readback": BalsamWireRequest.readback(GroupId("group"), BackendJobId("job")),
    }[operation]
    result = dispatch(request, FakeOperations(error=error))
    expected = "ambiguous" if operation in {"register", "submit", "rearm"} and not isinstance(error, BalsamControlRefusal) else "refused"
    assert result.status == expected
    assert decode_result(encode_result(result)) == result


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

    paths = _bootstrap_inputs(tmp_path)
    project, lock = paths["project_dir"], paths["lock"]
    python, uv, cache, prefix = paths["python"], paths["uv"], paths["cache_dir"], paths["prefix"]
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return __import__("subprocess").CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    built = build_prefix(project, lock, python=python, uv=uv, prefix=prefix, cache_dir=cache, runner=run)
    assert built["prefix"] == str(prefix)
    assert calls[0][0] == (str(uv), "sync", "--locked", "--no-editable", "--project", str(project), "--python", str(python))
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(prefix)
    assert calls[0][1]["env"]["UV_CACHE_DIR"] == str(cache)
    assert "XDG_CACHE_HOME" not in calls[0][1]["env"]
    assert build_prefix(project, lock, python=python, uv=uv, prefix=prefix, cache_dir=cache, runner=run) == built
    lock.write_bytes(b"changed")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(project, lock, python=python, uv=uv, prefix=prefix, cache_dir=cache, runner=run)


def test_prefix_manifest_binds_sources_inputs_and_executables(tmp_path: Path) -> None:
    """Source, selected inputs, and same-path tool replacement refuse reuse."""

    paths = _bootstrap_inputs(tmp_path)
    run = lambda argv, **kwargs: __import__("subprocess").CompletedProcess(argv, 0, stdout=b"", stderr=b"")
    build_prefix(**paths, runner=run)
    (paths["project_dir"] / "src" / "bridge.py").write_bytes(b"changed")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(**paths, runner=run)

    paths = _bootstrap_inputs(tmp_path / "other")
    build_prefix(**paths, runner=run)
    paths["python"].write_bytes(b"replacement")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(**paths, runner=run)

    paths = _bootstrap_inputs(tmp_path / "executor-drift")
    build_prefix(**paths, runner=run)
    (paths["project_dir"].parents[1] / "pyproject.toml").write_bytes(b"replacement")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(**paths, runner=run)

    paths = _bootstrap_inputs(tmp_path / "executor-source-drift")
    build_prefix(**paths, runner=run)
    (paths["project_dir"].parents[1] / "src" / "executor.py").write_bytes(b"replacement")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(**paths, runner=run)

    paths = _bootstrap_inputs(tmp_path / "uv-drift")
    build_prefix(**paths, runner=run)
    paths["uv"].write_bytes(b"replacement")
    with pytest.raises(BalsamControlRefusal, match="drifted"):
        build_prefix(**paths, runner=run)


def test_prefix_manifest_rejects_unrelated_lock_and_arbitrary_artifact(tmp_path: Path) -> None:
    """The selected lock is bound and no arbitrary artifact is accepted."""

    paths = _bootstrap_inputs(tmp_path)
    run = lambda argv, **kwargs: __import__("subprocess").CompletedProcess(argv, 0, stdout=b"", stderr=b"")
    build_prefix(**paths, runner=run)
    unrelated_lock = tmp_path / "unrelated.lock"
    unrelated_lock.write_bytes(b"other lock")
    with pytest.raises(BalsamControlRefusal, match="selected project uv.lock"):
        build_prefix(**{**paths, "lock": unrelated_lock}, runner=run)
    with pytest.raises(TypeError):
        build_prefix(**{**paths, "executor_artifact": tmp_path / "unrelated.whl"}, runner=run)


def test_prefix_manifest_rejects_home_and_unmanifested_prefix(tmp_path: Path) -> None:
    """Home fallback and adoption of an unknown prefix are both refused."""

    paths = _bootstrap_inputs(tmp_path)
    project, lock = paths["project_dir"], paths["lock"]
    python, uv, cache = paths["python"], paths["uv"], paths["cache_dir"]
    with pytest.raises(BalsamControlRefusal, match="outside the home"):
        prefix_manifest(project, lock, python=python, uv=uv, prefix=Path.home() / "control-prefix", cache_dir=cache)
    with pytest.raises(BalsamControlRefusal, match="cache_dir"):
        prefix_manifest(project, lock, python=python, uv=uv, prefix=paths["prefix"])
    with pytest.raises(BalsamControlRefusal, match="outside the home"):
        prefix_manifest(project, lock, python=python, uv=uv, prefix=paths["prefix"], cache_dir=Path.home() / "uv-cache")
    prefix = tmp_path / "unmanifested"
    prefix.mkdir()
    (prefix / "foreign").write_bytes(b"foreign")
    with pytest.raises(BalsamControlRefusal, match="unmanifested"):
        build_prefix(project, lock, python=python, uv=uv, prefix=prefix, cache_dir=cache)


def test_prefix_manifest_rejects_nested_outputs(tmp_path: Path) -> None:
    """Build outputs cannot become source-build inputs."""

    paths = _bootstrap_inputs(tmp_path)
    for field, nested in (("prefix", paths["project_dir"] / ".venv"), ("cache_dir", paths["project_dir"] / "cache")):
        with pytest.raises(BalsamControlRefusal, match="overlap"):
            prefix_manifest(**{**paths, field: nested})
    with pytest.raises(BalsamControlRefusal, match="overlap"):
        prefix_manifest(**{**paths, "cache_dir": paths["prefix"] / "cache"})


def test_bridge_validates_client_path_before_sdk_import(tmp_path: Path) -> None:
    """Missing client paths fail before optional SDK loading is attempted."""

    with pytest.raises(BalsamControlRefusal):
        load_sdk_operations(str(tmp_path / "missing-client"))
