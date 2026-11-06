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

    Parameters
    ----------
    context : object
        Object exposing every field of the toolkit ``AllocationContext``.
    allocation_id, backend_job_id : str
        Stable identities copied into the executor allocation record.
    observed_environment : mapping[str, str]
        Environment observed by the caller at attachment time.
    observed_hostname : str
        Hostname observed by the caller at attachment time.

    Returns
    -------
    dict
        Versioned JSON-shaped context snapshot.

    Raises
    ------
    ActivationRefusal
        If fields are missing, inconsistent, blank, or describe unsupported
        multi-slot or multi-node geometry.

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
    if type(context.allocation_id) is not str or not context.allocation_id:
        raise ActivationRefusal("allocation-context", "context allocation_id must be non-empty")
    if context.allocation_id != allocation_id:
        raise ActivationRefusal("allocation-context", "context and supplied allocation IDs disagree")
    if type(context.visibility_variable) is not str or not context.visibility_variable:
        raise ActivationRefusal("allocation-context", "visibility_variable must be non-empty")
    if type(context.deadline_guard_min) is not int or context.deadline_guard_min < 0:
        raise ActivationRefusal("allocation-context", "deadline_guard_min must be a non-negative integer")
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


def bind_allocation_context(
    allocation_path: str | Path,
    output_path: str | Path,
    context: Any,
    *,
    observed_environment: Mapping[str, str],
    observed_hostname: str,
) -> Path:
    """Write the validated full context into the allocation consumed by dispatch.

    Parameters
    ----------
    allocation_path : str or pathlib.Path
        Existing executor allocation JSON record.
    output_path : str or pathlib.Path
        Distinct absolute allocation record passed to the executor CLI.
    context : object
        Allocation-context object carrying every attach field.
    observed_environment : mapping[str, str]
        Environment observed by the caller before dispatch.
    observed_hostname : str
        Hostname observed by the caller before dispatch.

    Returns
    -------
    pathlib.Path
        The newly published allocation record used by the route.

    Raises
    ------
    ActivationRefusal
        If the source record, context identity, or output path is invalid.
    """

    source = Path(allocation_path)
    output = Path(output_path)
    if not source.is_absolute() or not output.is_absolute():
        raise ActivationRefusal("allocation-path", "allocation paths must be absolute")
    if source.resolve() == output.resolve():
        raise ActivationRefusal("allocation-path", "bound allocation output must be a distinct path")
    payload = _strict_json(source.read_text(encoding="utf-8"), "allocation")
    expected = {"schema", "allocation_id", "backend_job_id", "facts"}
    if set(payload) != expected or payload["schema"] != "tpen-executor/configured-allocation-v1":
        raise ActivationRefusal("allocation-shape", "allocation record has an unexpected schema")
    if type(payload["allocation_id"]) is not str or type(payload["backend_job_id"]) is not str:
        raise ActivationRefusal("allocation-shape", "allocation identities must be strings")
    if not isinstance(payload["facts"], Mapping):
        raise ActivationRefusal("allocation-shape", "allocation facts must be a mapping")
    snapshot = allocation_context_payload(
        context,
        allocation_id=payload["allocation_id"],
        backend_job_id=payload["backend_job_id"],
        observed_environment=observed_environment,
        observed_hostname=observed_hostname,
    )
    facts = dict(payload["facts"])
    binding = dict(facts.get("binding", {}))
    operational_environment = dict(snapshot["context"]["environment"])
    if snapshot["context"]["visibility_values"]:
        operational_environment[snapshot["context"]["visibility_variable"]] = ",".join(
            snapshot["context"]["visibility_values"]
        )
    binding["environment"] = operational_environment
    binding["visibility"] = list(snapshot["context"]["visibility_values"])
    facts["binding"] = binding
    facts["attached_context"] = snapshot
    payload["facts"] = facts
    _atomic_write(output, canonical_bytes(payload) + b"\n")
    return output


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one input file without resolving or rewriting it.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing file to hash.

    Returns
    -------
    str
        Lowercase SHA-256 digest.

    Raises
    ------
    OSError
        If the file cannot be read.
    """

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


def retire_attached_generation(directory: str | Path, error: BaseException) -> None:
    """Retire current attached sidecars and publish a failed-generation diagnostic.

    Parameters
    ----------
    directory : str or pathlib.Path
        Absolute directory containing the current attached handoff files.
    error : BaseException
        The selected-generation failure to preserve in the diagnostic record.

    Returns
    -------
    None
        Current sidecars are retired and a diagnostic is published.

    Raises
    ------
    ActivationRefusal
        If ``directory`` is not absolute.
    OSError
        If evidence cannot be retired or published.
    """

    root = Path(directory)
    if not root.is_absolute():
        raise ActivationRefusal("handoff-path", "attached handoff directory must be absolute")
    token = hashlib.sha256(repr(error).encode()).hexdigest()
    _retire(root / "executor_input.json", token)
    _retire(root / "configured-runs.json", token)
    _retire(root / "executor_input_diagnostics.json", token)
    _atomic_write(
        root / "executor_input_diagnostics.json",
        canonical_bytes(
            {
                "schema": "tpen-experiments/attached-input-diagnostics-v1",
                "code": getattr(error, "code", "translation-failed"),
                "reason": str(error),
            }
        ),
    )


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

    Parameters
    ----------
    plan : StagePlanV2
        Validated ordered producer plan.
    effective_config, source_config : str or pathlib.Path
        Absolute effective and source YAML paths.
    profile : str, default=PROFILE_NAME
        Declared core profile identity.
    overrides_by_task : mapping[str, sequence[str]], optional
        Exact per-task override lists to bind into the group.
    group_id : str, optional
        Explicit opaque group identity.
    metadata : mapping, optional
        Additional provenance carried in the opaque program reference.

    Returns
    -------
    tuple of dict, dict, AttachedHandoff
        Group wire mapping, configured-runs wire mapping, and their identity.

    Raises
    ------
    ActivationRefusal
        If a path, task, override, or group member cannot be represented.

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

    Parameters
    ----------
    directory : str or pathlib.Path
        Absolute handoff directory.
    plan : StagePlanV2
        Validated ordered producer plan.
    effective_config, source_config, profile, overrides_by_task, group_id, metadata
        Inputs forwarded to :func:`build_attached_inputs`.

    Returns
    -------
    AttachedHandoff
        Current sidecar paths and bound identity.

    Raises
    ------
    ActivationRefusal
        If translation fails; current sidecars are retired and a diagnostic is
        published before the refusal is re-raised.

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
        retire_attached_generation(root, error)
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
    """Parse a profile JSON verdict; process exit status is not the verdict.

    Parameters
    ----------
    stdout : str
        Exactly one profile-check JSON object.
    expected_profile : str, default=PROFILE_NAME
        Requested profile identity.
    returncode : int, default=0
        Process status recorded for provenance only.

    Returns
    -------
    ProfileVerdict
        Typed profile response.

    Raises
    ------
    ActivationRefusal
        If the JSON schema, profile, verdict, or refusal shape is invalid.
    """

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
    overrides: Sequence[str] = (),
    cwd: str | Path | None = None,
) -> ProfileVerdict:
    """Run the prepared science profile checker and require supported JSON.

    Parameters
    ----------
    science_python : str or pathlib.Path
        Absolute prepared science Python executable.
    config : str or pathlib.Path
        Absolute effective config path.
    profile : str, default=PROFILE_NAME
        Profile requested from the checker.
    overrides : sequence of str, optional
        Unresolved dot-list overrides to merge with ``config`` in the science
        environment before the public profile checker is invoked.
    cwd : str or pathlib.Path, optional
        Checker working directory.

    Returns
    -------
    ProfileVerdict
        Supported typed profile verdict.

    Raises
    ------
    ActivationRefusal
        If the checker output is malformed, mismatched, or refused.
    """

    python = _absolute_config(str(science_python), "science_python")
    config_path = _absolute_config(str(config), "config")
    command = [python, "-m", "tpen.checkpoint.profiles", config_path, "--profile", profile]
    if overrides:
        compose_script = """
import json
import subprocess
import sys
from omegaconf import OmegaConf

base = OmegaConf.load(sys.argv[1])
merged = OmegaConf.merge(base, OmegaConf.from_dotlist(json.loads(sys.argv[4])))
OmegaConf.save(config=merged, f=sys.argv[2], resolve=False)
result = subprocess.run(
    [sys.executable, "-m", "tpen.checkpoint.profiles", sys.argv[2], "--profile", sys.argv[3]],
    capture_output=True,
    text=True,
    check=False,
)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
raise SystemExit(result.returncode)
"""
        with tempfile.TemporaryDirectory(
            prefix=".tpen-profile-",
            dir=None if cwd is None else str(cwd),
        ) as temporary:
            merged_path = Path(temporary) / "merged.yaml"
            script_path = Path(temporary) / "compose_and_check.py"
            script_path.write_text(compose_script, encoding="utf-8")
            result = subprocess.run(
                [python, str(script_path), config_path, str(merged_path), profile, json.dumps(list(overrides))],
                cwd=None if cwd is None else str(cwd),
                capture_output=True,
                text=True,
                check=False,
            )
    else:
        result = subprocess.run(
            command,
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


def _validate_attached_sidecars(handoff: AttachedHandoff) -> None:
    """Refuse sidecars whose P10 binding differs from a fresh handoff."""

    try:
        group = _strict_json(Path(handoff.executor_input).read_text(encoding="utf-8"), "executor input")
        table = _strict_json(Path(handoff.configured_runs).read_text(encoding="utf-8"), "configured runs")
    except OSError as error:
        raise ActivationRefusal("handoff-missing", f"attached sidecar could not be read: {error}") from error
    if payload_digest(group) != handoff.group_digest:
        raise ActivationRefusal("handoff-group-mismatch", "executor input no longer matches the bound group digest")
    if table.get("schema") != CONFIGURED_RUNS_SCHEMA or table.get("group_digest") != handoff.group_digest:
        raise ActivationRefusal("handoff-runs-mismatch", "configured-runs sidecar is not bound to the fresh group")
    members = group.get("members")
    runs = table.get("runs")
    if not isinstance(members, list) or not isinstance(runs, list) or len(members) != len(runs):
        raise ActivationRefusal("handoff-shape", "attached sidecars have different member counts")
    if tuple(str(member.get("task_id")) for member in members) != handoff.task_ids:
        raise ActivationRefusal("handoff-task-mismatch", "executor input task IDs differ from the fresh handoff")
    for member, run in zip(members, runs, strict=True):
        if not isinstance(member, Mapping) or not isinstance(run, Mapping):
            raise ActivationRefusal("handoff-shape", "attached sidecars contain a non-object member")
        task_id = member.get("task_id")
        program = member.get("program")
        config_ref = program.get("config_ref") if isinstance(program, Mapping) else None
        binding = config_ref.get("attached_activation") if isinstance(config_ref, Mapping) else None
        if not isinstance(binding, Mapping):
            raise ActivationRefusal("handoff-binding-missing", f"task {task_id!r} has no attached P10 binding")
        expected = {
            "effective_config": handoff.effective_config,
            "effective_config_sha256": handoff.effective_config_sha256,
            "source_config": handoff.source_config,
            "source_config_sha256": handoff.source_config_sha256,
            "profile": handoff.profile,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ActivationRefusal("handoff-binding-mismatch", f"task {task_id!r} binding changed")
        if run.get("id") != task_id or run.get("config") != handoff.effective_config:
            raise ActivationRefusal("handoff-run-mismatch", f"task {task_id!r} configured run changed")
        bound_overrides = binding.get("overrides")
        if not isinstance(bound_overrides, list) or run.get("overrides") != bound_overrides:
            raise ActivationRefusal("handoff-overrides-mismatch", f"task {task_id!r} overrides changed")
    if file_sha256(handoff.effective_config) != handoff.effective_config_sha256:
        raise ActivationRefusal("effective-config-mismatch", "effective config bytes differ from the fresh handoff")
    if file_sha256(handoff.source_config) != handoff.source_config_sha256:
        raise ActivationRefusal("source-config-mismatch", "source config bytes differ from the fresh handoff")


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
    """Invoke the installed P9c CLI in an absolute executor environment.

    Parameters
    ----------
    executor_python : str or pathlib.Path
        Absolute Python executable containing the installed executor package.
    handoff : AttachedHandoff
        Fresh handoff identity to validate against both sidecars.
    allocation, worker_config, store : str or pathlib.Path
        Absolute allocation record, worker YAML, and fresh store paths.
    dispatcher : {"local", "parsl"}, default="parsl"
        Executor dispatcher backend.
    timeout : float, optional
        Positive child/evidence timeout.
    cwd : str or pathlib.Path, optional
        Working directory for the executor subprocess.

    Returns
    -------
    ExecutorDispatchResult
        Captured CLI result and any decoded manifest.

    Raises
    ------
    ActivationRefusal
        If the sidecars no longer match the bound P10 program.
    """

    python = _absolute_config(str(executor_python), "executor_python")
    _validate_attached_sidecars(handoff)
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


def validate_child_context_observation(
    allocation_path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Compare child-owned context observations with the validated binding.

    Parameters
    ----------
    allocation_path : str or pathlib.Path
        Bound allocation record containing the attached context snapshot.
    manifest : mapping
        Published executor manifest containing the completed attempt outputs.

    Returns
    -------
    None
        The function returns only when every child reports the validated
        visibility and explicit context-environment values.

    Raises
    ------
    ActivationRefusal
        If the bound snapshot, child observation evidence, or observed values
        cannot be trusted.  The named refusal preserves the completed child
        evidence while preventing a success activation receipt.

    The observation file is child-owned evidence written by the executor
    runner after any supported ``SourceScript`` setup. Therefore this
    comparison is intentionally made after executor dispatch and before the
    root reader can publish success.
    """

    try:
        allocation = _strict_json(Path(allocation_path).read_text(encoding="utf-8"), "allocation")
        facts = allocation.get("facts")
        snapshot = facts.get("attached_context") if isinstance(facts, Mapping) else None
        context = snapshot.get("context") if isinstance(snapshot, Mapping) else None
        if not isinstance(context, Mapping):
            raise ValueError("bound allocation has no attached context snapshot")
        visibility_variable = context.get("visibility_variable")
        visibility_values = context.get("visibility_values")
        environment = context.get("environment")
        if (
            type(visibility_variable) is not str
            or not visibility_variable
            or not isinstance(visibility_values, list)
            or not isinstance(environment, Mapping)
            or any(type(key) is not str or type(value) is not str for key, value in environment.items())
        ):
            raise ValueError("attached context snapshot has an invalid environment shape")
        expected = dict(environment)
        if visibility_values:
            if any(type(value) is not str or not value for value in visibility_values):
                raise ValueError("attached context snapshot has invalid visibility values")
            expected[visibility_variable] = ",".join(visibility_values)

        attempts = manifest.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("executor manifest has no attempts to observe")
        for attempt in attempts:
            output_dir = attempt.get("output_dir") if isinstance(attempt, Mapping) else None
            if type(output_dir) is not str or not output_dir:
                raise ValueError("executor manifest attempt has no output directory")
            observation = _strict_json(
                (Path(output_dir) / "executor-child-observation.json").read_text(encoding="utf-8"),
                "child observation",
            )
            if set(observation) != {
                "schema",
                "cuda_visible_devices",
                "environment",
                "thread_settings",
                "hostname",
                "pid",
            } or observation["schema"] != "tpen-executor/child-observation-v1":
                raise ValueError("child observation has an unexpected schema")
            observed = observation["environment"]
            if not isinstance(observed, Mapping) or any(
                type(key) is not str or type(value) is not str for key, value in observed.items()
            ):
                raise ValueError("child observation has no string-valued environment")
            thread_settings = observation["thread_settings"]
            if not isinstance(thread_settings, Mapping) or set(thread_settings) != {
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            } or any(value is not None and type(value) is not str for value in thread_settings.values()):
                raise ValueError("child observation has invalid thread settings")
            if observation["cuda_visible_devices"] is not None and type(observation["cuda_visible_devices"]) is not str:
                raise ValueError("child observation has invalid CUDA visibility")
            if type(observation["hostname"]) is not str or not observation["hostname"]:
                raise ValueError("child observation has invalid hostname")
            if type(observation["pid"]) is not int or isinstance(observation["pid"], bool) or observation["pid"] <= 0:
                raise ValueError("child observation has invalid PID")
            attached = facts.get("attached_context") if isinstance(facts, Mapping) else None
            observed_snapshot = attached.get("observed") if isinstance(attached, Mapping) else None
            if not isinstance(observed_snapshot, Mapping):
                raise ValueError("bound allocation has no observed context snapshot")
            expected_hostname = observed_snapshot.get("hostname")
            if type(expected_hostname) is not str or not expected_hostname:
                raise ValueError("bound allocation has no observed hostname")
            if observation["hostname"] != expected_hostname:
                raise ActivationRefusal(
                    "child-context-mismatch",
                    f"child hostname differs from validated allocation: expected {expected_hostname!r}, observed {observation['hostname']!r}",
                )
            mismatches = {
                key: {"expected": value, "observed": observed.get(key)}
                for key, value in expected.items()
                if observed.get(key) != value
            }
            if mismatches:
                raise ActivationRefusal(
                    "child-context-mismatch",
                    f"child context differs from validated allocation: {mismatches}",
                )
    except ActivationRefusal:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise ActivationRefusal("child-context-observation", str(error)) from error


def write_activation_receipt(
    manifest_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_task_ids: Sequence[str],
    inspect_run: Callable[[str | Path], Any] | None = None,
    profile: str = PROFILE_NAME,
) -> ResultSet:
    """Read completed core evidence and write the P10 activation receipt.

    Parameters
    ----------
    manifest_path, receipt_path : str or pathlib.Path
        Published executor manifest and durable activation receipt paths.
    expected_task_ids : sequence of str
        Exact ordered task identities expected from the selected generation.
    inspect_run : callable, optional
        Public core inspector or a test double receiving each fenced core path.
    profile : str, default=PROFILE_NAME
        Profile identity recorded in the receipt.

    Returns
    -------
    ResultSet
        Reader result after the completed-verdict gate.

    Raises
    ------
    ManifestRefusal, OSError, ValueError
        If the manifest, public evidence, or receipt cannot be accepted.
    """

    manifest = Path(manifest_path)
    receipt = Path(receipt_path)
    expected = tuple(str(item) for item in expected_task_ids)
    try:
        inspector = inspect_run
        if inspector is None:
            from .run_results import _default_inspect_run

            inspector = _default_inspect_run

        def completed_inspector(path: str | Path) -> Any:
            evidence = inspector(path)
            if isinstance(evidence, Mapping):
                verdict = evidence.get("verdict")
                run_status = evidence.get("run_status")
            else:
                verdict = getattr(evidence, "verdict", None)
                run_status = getattr(evidence, "run_status", None)
            if verdict != "completed" or run_status != "completed":
                raise ManifestRefusal(
                    "core-run-incomplete",
                    "P10 activation requires public core verdict and run_status completed",
                )
            return evidence

        result = read(manifest, inspect_run=completed_inspector)
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
    """Preserve a named pre-reader activation failure as durable evidence.

    Parameters
    ----------
    receipt_path : str or pathlib.Path
        Destination for the failed activation receipt.
    code, reason : str
        Named refusal code and human-readable reason.
    expected_task_ids : sequence of str, optional
        Task identities selected before the failure.
    profile : str, default=PROFILE_NAME
        Profile identity recorded in the receipt.
    manifest_path : str or pathlib.Path, optional
        Existing manifest whose digest should be retained.

    Returns
    -------
    None
        The failure receipt is atomically published.

    Raises
    ------
    OSError
        If the receipt path cannot be written.
    """

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
    "bind_allocation_context",
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
    "validate_child_context_observation",
    "retire_attached_generation",
    "write_activation_receipt",
    "write_activation_failure_receipt",
    "write_attached_handoff",
]
