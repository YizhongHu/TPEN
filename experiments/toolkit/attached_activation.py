"""Study-side handoff and activation for an attached executor allocation.

This module is deliberately independent of :mod:`tpen_executor`.  It owns the
plain-data handoff at the experiment boundary, invokes the installed executor
CLI in a separate process, and reads the resulting manifest through the shared
toolkit reader.  The default experiment launchers do not call this module;
callers must opt into the attached route explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .dispatch import StagePlanV2
from .executor_inputs import canonical_bytes, payload_digest, ready_group
from .run_results import ManifestRefusal, ResultSet, read


ATTACHED_ACTIVATION_SCHEMA = "tpen-experiments/attached-activation-v1"
PROFILE_CHECK_SCHEMA = "tpen.profile-check/v1"
PROFILE_NAME = "tpen.native-recovery/v1"
CONFIGURED_RUNS_SCHEMA = "tpen-executor/configured-runs-v1"


class ActivationRefusal(ValueError):
    """A named refusal before or after an attached activation attempt."""

    def __init__(self, code: str, reason: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("activation refusal code must be non-empty")
        if type(reason) is not str or not reason:
            raise TypeError("activation refusal reason must be non-empty")
        self.code = code
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class AttachedHandoff:
    """The current files for one fresh attached-input generation."""

    directory: str
    executor_input: str
    configured_runs: str
    group_digest: str
    task_ids: tuple[str, ...]
    effective_config: str
    effective_config_sha256: str
    source_config: str
    source_config_sha256: str
    profile: str


@dataclass(frozen=True, slots=True)
class ProfileVerdict:
    """The JSON verdict returned by the core profile checker."""

    profile: str
    verdict: str
    state_owners: tuple[str, ...]
    refusal: Mapping[str, str] | None
    returncode: int


@dataclass(frozen=True, slots=True)
class ExecutorDispatchResult:
    """Raw result of one public executor CLI subprocess."""

    returncode: int
    manifest: Mapping[str, Any] | None
    stdout: str
    stderr: str


def allocation_context_payload(
    context: Any,
    *,
    allocation_id: str,
    backend_job_id: str,
    observed_environment: Mapping[str, str],
    observed_hostname: str,
) -> dict[str, Any]:
    """Snapshot the bounded attached-allocation context without dropping fields.

    This is intentionally a data adapter rather than a placement claim.  The
    P10 route admits only one worker visibility value and one node; physical
    slot identity remains an owner of the placement qualification item.
    """

    names = (
        "allocation_id",
        "visibility_variable",
        "visibility_values",
        "run_root",
        "deadline",
        "deadline_env_var",
        "deadline_guard_min",
        "environment",
        "nodes_per_block",
    )
    missing = [name for name in names if not hasattr(context, name)]
    if missing:
        raise ActivationRefusal("allocation-context", f"allocation context omits fields: {missing}")
    visibility = tuple(context.visibility_values)
    if any(type(item) is not str or not item for item in visibility):
        raise ActivationRefusal("allocation-context", "visibility values must be non-empty strings")
    if len(visibility) > 1:
        raise ActivationRefusal("multi-slot-context", "attached P10 route accepts one visibility slot")
    if context.nodes_per_block not in (None, 1):
        raise ActivationRefusal("multi-node-context", "attached P10 route accepts one node per block")
    if type(allocation_id) is not str or not allocation_id or type(backend_job_id) is not str or not backend_job_id:
        raise ActivationRefusal("allocation-context", "allocation and backend identities must be non-empty")
    if type(observed_hostname) is not str or not observed_hostname:
        raise ActivationRefusal("allocation-observation", "observed hostname must be non-empty")
    if not isinstance(context.environment, Mapping):
        raise ActivationRefusal("allocation-context", "context environment must be a mapping")
    if any(type(key) is not str or type(value) is not str for key, value in context.environment.items()):
        raise ActivationRefusal("allocation-context", "context environment must be string-valued")
    if not isinstance(observed_environment, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in observed_environment.items()
    ):
        raise ActivationRefusal("allocation-observation", "observed environment must be string-valued")
    return {
        "schema": "tpen-experiments/attached-allocation-context-v1",
        "allocation_id": allocation_id,
        "backend_job_id": backend_job_id,
        "context": {
            "allocation_id": context.allocation_id,
            "visibility_variable": context.visibility_variable,
            "visibility_values": list(visibility),
            "run_root": context.run_root,
            "deadline": context.deadline,
            "deadline_env_var": context.deadline_env_var,
            "deadline_guard_min": context.deadline_guard_min,
            "environment": dict(context.environment),
            "nodes_per_block": context.nodes_per_block,
        },
        "observed": {
            "hostname": observed_hostname,
            "environment": dict(observed_environment),
        },
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one input file without resolving or rewriting it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(text: str, name: str) -> dict[str, Any]:
    """Decode exactly one JSON object and reject duplicate members."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActivationRefusal("duplicate-json-key", f"{name} repeats {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except ActivationRefusal:
        raise
    except (TypeError, ValueError) as error:
        raise ActivationRefusal("malformed-json", f"{name} is not one JSON object: {error}") from error
    if type(value) is not dict:
        raise ActivationRefusal("malformed-json", f"{name} must be one JSON object")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish bytes by same-directory temporary-file replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _retire(path: Path, token: str) -> None:
    """Move a superseded current artifact aside without deleting evidence."""

    if not path.exists():
        return
    candidate = path.with_name(f"{path.stem}.superseded-{token}{path.suffix}")
    serial = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.superseded-{token}-{serial}{path.suffix}")
        serial += 1
    os.replace(path, candidate)


def _absolute_config(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ActivationRefusal("config-path", f"{name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ActivationRefusal("config-path", f"{name} must be absolute")
    return str(path)


def _attached_program(
    member: Mapping[str, Any], binding: Mapping[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    """Bind effective-config identity into an opaque program reference."""

    program = dict(member["program"])
    config_ref = dict(program["config_ref"])
    attached = dict(binding)
    attached["overrides"] = list(overrides)
    config_ref["attached_activation"] = attached
    program["config_ref"] = config_ref
    program["content_digest"] = payload_digest(config_ref)
    return program


def build_attached_inputs(
    plan: StagePlanV2,
    *,
    effective_config: str | Path,
    source_config: str | Path,
    profile: str = PROFILE_NAME,
    overrides_by_task: Mapping[str, Sequence[str]] | None = None,
    group_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], AttachedHandoff]:
    """Build a complete attached group and its configured-run sidecar.

    The group is built in the root project and then augmented with the exact
    effective/source config paths and hashes before its order-sensitive group
    identity is computed.  The sidecar mirrors that same ordered identity; it
    is never derived from a prefix or from a separately sorted task list.
    """

    if not isinstance(plan, StagePlanV2):
        raise TypeError("plan must be StagePlanV2")
    plan.validate()
    if not plan.tasks:
        raise ActivationRefusal("empty-group", "attached activation requires at least one task")
    effective = _absolute_config(str(effective_config), "effective_config")
    source = _absolute_config(str(source_config), "source_config")
    effective_hash = file_sha256(effective)
    source_hash = file_sha256(source)
    binding = {
        "effective_config": effective,
        "effective_config_sha256": effective_hash,
        "source_config": source,
        "source_config_sha256": source_hash,
        "profile": profile,
    }
    if metadata:
        binding["metadata"] = dict(metadata)
    group = ready_group(plan, group_id=group_id)
    members: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    overrides_by_task = {} if overrides_by_task is None else overrides_by_task
    for member, task in zip(group["members"], plan.tasks, strict=True):
        task_id = str(member["task_id"])
        overrides = tuple(str(item) for item in overrides_by_task.get(task_id, task.params.get("executor_overrides", ())))
        if any(not item for item in overrides):
            raise ActivationRefusal("empty-override", f"task {task_id!r} has an empty override")
        attached_member = dict(member)
        attached_member["program"] = _attached_program(member, binding, overrides)
        members.append(attached_member)
        runs.append({
            "id": task_id,
            "config": effective,
            "overrides": list(overrides),
        })
    group = dict(group)
    group["members"] = members
    group["execution_plan_digest"] = payload_digest({
        "base": group["execution_plan_digest"],
        "attached": binding,
        "task_ids": [member["task_id"] for member in members],
    })
    group_digest = payload_digest(group)
    table = {
        "schema": CONFIGURED_RUNS_SCHEMA,
        "group_digest": group_digest,
        "runs": runs,
    }
    handoff = AttachedHandoff(
        directory="",
        executor_input="",
        configured_runs="",
        group_digest=group_digest,
        task_ids=tuple(member["task_id"] for member in members),
        effective_config=effective,
        effective_config_sha256=effective_hash,
        source_config=source,
        source_config_sha256=source_hash,
        profile=profile,
    )
    return group, table, handoff


def write_attached_handoff(
    directory: str | Path,
    plan: StagePlanV2,
    *,
    effective_config: str | Path,
    source_config: str | Path,
    profile: str = PROFILE_NAME,
    overrides_by_task: Mapping[str, Sequence[str]] | None = None,
    group_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AttachedHandoff:
    """Write both attached input files for a fresh generation.

    Existing current files are superseded by digest-named siblings.  A failed
    translation retires stale current inputs and leaves a diagnostic artifact,
    so an earlier representable prefix cannot be dispatched accidentally.
    """

    root = Path(directory)
    if not root.is_absolute():
        raise ActivationRefusal("handoff-path", "attached handoff directory must be absolute")
    try:
        group, table, handoff = build_attached_inputs(
            plan,
            effective_config=effective_config,
            source_config=source_config,
            profile=profile,
            overrides_by_task=overrides_by_task,
            group_id=group_id,
            metadata=metadata,
        )
    except Exception as error:
        token = hashlib.sha256(repr(error).encode()).hexdigest()
        _retire(root / "executor_input.json", token)
        _retire(root / "configured-runs.json", token)
        _retire(root / "executor_input_diagnostics.json", token)
        _atomic_write(
            root / "executor_input_diagnostics.json",
            canonical_bytes({
                "schema": "tpen-experiments/attached-input-diagnostics-v1",
                "code": getattr(error, "code", "translation-failed"),
                "reason": str(error),
            }),
        )
        raise
    token = handoff.group_digest
    _retire(root / "executor_input.json", token)
    _retire(root / "configured-runs.json", token)
    _retire(root / "executor_input_diagnostics.json", token)
    _atomic_write(root / "executor_input.json", canonical_bytes(group) + b"\n")
    _atomic_write(root / "configured-runs.json", canonical_bytes(table) + b"\n")
    return AttachedHandoff(
        directory=str(root),
        executor_input=str(root / "executor_input.json"),
        configured_runs=str(root / "configured-runs.json"),
        **{field: getattr(handoff, field) for field in handoff.__dataclass_fields__ if field not in {"directory", "executor_input", "configured_runs"}},
    )


def parse_profile_verdict(stdout: str, *, expected_profile: str = PROFILE_NAME, returncode: int = 0) -> ProfileVerdict:
    """Parse a profile JSON verdict; process exit status is not the verdict."""

    payload = _strict_json(stdout, "profile stdout")
    expected = {"schema", "profile", "verdict", "state_owners", "refusal"}
    if set(payload) != expected or payload["schema"] != PROFILE_CHECK_SCHEMA:
        raise ActivationRefusal("profile-unavailable", "profile response has an unexpected schema")
    if payload["profile"] != expected_profile:
        raise ActivationRefusal("profile-mismatch", "profile response does not match the requested profile")
    if payload["verdict"] not in {"supported", "refused"}:
        raise ActivationRefusal("profile-unavailable", "profile response has an unknown verdict")
    owners = payload["state_owners"]
    if isinstance(owners, (str, bytes, bytearray)) or not isinstance(owners, list) or any(type(item) is not str for item in owners):
        raise ActivationRefusal("profile-unavailable", "profile state_owners is malformed")
    refusal = payload["refusal"]
    if payload["verdict"] == "supported" and refusal is not None:
        raise ActivationRefusal("profile-unavailable", "supported profile has a refusal")
    if payload["verdict"] == "refused" and not isinstance(refusal, dict):
        raise ActivationRefusal("profile-unavailable", "refused profile lacks a refusal")
    return ProfileVerdict(expected_profile, payload["verdict"], tuple(sorted(set(owners))), refusal, returncode)


def check_profile(
    science_python: str | Path,
    config: str | Path,
    *,
    profile: str = PROFILE_NAME,
    cwd: str | Path | None = None,
) -> ProfileVerdict:
    """Run the prepared science profile checker and require supported JSON."""

    python = _absolute_config(str(science_python), "science_python")
    config_path = _absolute_config(str(config), "config")
    result = subprocess.run(
        [python, "-m", "tpen.checkpoint.profiles", config_path, "--profile", profile],
        cwd=None if cwd is None else str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    verdict = parse_profile_verdict(result.stdout, expected_profile=profile, returncode=result.returncode)
    if verdict.verdict != "supported":
        reason = "profile refused"
        if verdict.refusal is not None:
            reason = str(verdict.refusal.get("reason", reason))
        raise ActivationRefusal("profile-refused", reason)
    return verdict


def run_executor_cli(
    executor_python: str | Path,
    handoff: AttachedHandoff,
    *,
    allocation: str | Path,
    worker_config: str | Path,
    store: str | Path,
    dispatcher: str = "parsl",
    timeout: float | None = None,
    cwd: str | Path | None = None,
) -> ExecutorDispatchResult:
    """Invoke the installed P9c CLI in an absolute executor environment."""

    python = _absolute_config(str(executor_python), "executor_python")
    command = [
        python,
        "-m",
        "tpen_executor",
        "dispatch",
        "--input",
        handoff.executor_input,
        "--runs",
        handoff.configured_runs,
        "--allocation",
        _absolute_config(str(allocation), "allocation"),
        "--worker-config",
        _absolute_config(str(worker_config), "worker_config"),
        "--store",
        _absolute_config(str(store), "store"),
        "--dispatcher",
        dispatcher,
    ]
    if timeout is not None:
        command.extend(["--timeout", str(timeout)])
    result = subprocess.run(command, cwd=None if cwd is None else str(cwd), capture_output=True, text=True, check=False)
    manifest = None
    if result.stdout.strip():
        try:
            manifest = _strict_json(result.stdout, "executor stdout")
        except ActivationRefusal:
            if result.returncode in {0, 3}:
                raise
    return ExecutorDispatchResult(result.returncode, manifest, result.stdout, result.stderr)


def write_activation_receipt(
    manifest_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_task_ids: Sequence[str],
    inspect_run: Callable[[str | Path], Any] | None = None,
    profile: str = PROFILE_NAME,
) -> ResultSet:
    """Read a committed manifest in the root environment and write its receipt."""

    manifest = Path(manifest_path)
    receipt = Path(receipt_path)
    expected = tuple(str(item) for item in expected_task_ids)
    try:
        result = read(manifest, inspect_run=inspect_run)
        actual = tuple(chain.task_id for chain in result.chains)
        if actual != expected:
            raise ManifestRefusal("task-set-mismatch", f"reader chains {actual!r} differ from expected {expected!r}")
        if not result.chains or any(not chain.segments for chain in result.chains):
            raise ManifestRefusal("empty-chain", "reader returned no nonempty committed chain")
    except (ManifestRefusal, OSError, ValueError) as error:
        payload = {
            "schema": ATTACHED_ACTIVATION_SCHEMA,
            "status": "failed",
            "profile": profile,
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
            "expected_task_ids": list(expected),
            "refusal": {"code": getattr(error, "code", "reader-failed"), "reason": str(error)},
        }
        _atomic_write(receipt, canonical_bytes(payload) + b"\n")
        raise
    payload = {
        "schema": ATTACHED_ACTIVATION_SCHEMA,
        "status": "activated",
        "profile": profile,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "task_ids": list(expected),
        "chains": [
            {
                "task_id": chain.task_id,
                "segments": [
                    {
                        "attempt_id": segment.attempt_id,
                        "checkpoint_dir": segment.checkpoint_dir,
                        "next_iteration": segment.next_iteration,
                        "completed_updates": segment.completed_updates,
                    }
                    for segment in chain.segments
                ],
            }
            for chain in result.chains
        ],
        "preserved_attempts": len(result.preserved_attempts),
    }
    _atomic_write(Path(receipt), canonical_bytes(payload) + b"\n")
    return result


def write_activation_failure_receipt(
    receipt_path: str | Path,
    *,
    code: str,
    reason: str,
    expected_task_ids: Sequence[str] = (),
    profile: str = PROFILE_NAME,
    manifest_path: str | Path | None = None,
) -> None:
    """Preserve a named pre-reader activation failure as durable evidence."""

    payload = {
        "schema": ATTACHED_ACTIVATION_SCHEMA,
        "status": "failed",
        "profile": profile,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": (
            file_sha256(manifest_path) if manifest_path is not None and Path(manifest_path).is_file() else None
        ),
        "expected_task_ids": [str(item) for item in expected_task_ids],
        "refusal": {"code": str(code), "reason": str(reason)},
    }
    _atomic_write(Path(receipt_path), canonical_bytes(payload) + b"\n")


__all__ = [
    "ATTACHED_ACTIVATION_SCHEMA",
    "CONFIGURED_RUNS_SCHEMA",
    "ActivationRefusal",
    "allocation_context_payload",
    "AttachedHandoff",
    "ExecutorDispatchResult",
    "PROFILE_CHECK_SCHEMA",
    "PROFILE_NAME",
    "ProfileVerdict",
    "build_attached_inputs",
    "check_profile",
    "file_sha256",
    "parse_profile_verdict",
    "run_executor_cli",
    "write_activation_receipt",
    "write_activation_failure_receipt",
    "write_attached_handoff",
]
