"""Attempt-free dispatch seam between planning, admission, and execution.

This module is additive: it introduces a second, parallel surface next to the
legacy ``specs``/``execution``/``executors`` modules rather than changing them.
The two surfaces coexist, so every name here is deliberately distinct from its
legacy counterpart (``StagePlanV2`` beside ``StagePlan``, ``DispatchRecord``
beside ``ExecutionRecord``).

The seam splits three responsibilities that the legacy surface fused together:

Planning
    A planner issues a :class:`StagePlanV2` of :class:`LogicalTaskSpec` rows.
    A plan is *attempt-free*: it says what science should happen, never how many
    times it has been tried. Each row carries an **opaque** ``logical_task_id``
    that the planner owns.

Admission
    :func:`admit_tasks` takes a plan, a subset of its rows, and one concrete
    ``argv`` per runtime, and mints :class:`DispatchSpec` objects. The
    ``admission_id`` is the claim scope: retrying means minting a *fresh*
    ``admission_id``, which yields fresh ``attempt_id`` values. Nothing else in
    the system invents attempts.

Execution
    A :class:`DispatchExecutor` receives ready-made dispatches and returns
    :class:`DispatchRecord` acceptance receipts. Executors never plan, never
    retry, never mutate ``argv``, and never infer completion from exit status.

Identity policy
---------------
``logical_task_id`` is opaque. :func:`logical_task_id_from_parts` is a *naming
convention* offered to planners; it is enforced nowhere. No code in this module
re-derives or validates an identifier against it, and none ever should — that
re-derivation is exactly what made the legacy ``task_id``/``attempt_id`` coupling
impossible to retry cleanly.

Notes
-----
Reusable structures (:class:`~experiments.toolkit.resources.ResourceSpec`,
:class:`~experiments.toolkit.specs.CompletionSpec`, the JSON helpers, and the
allocation deadline helper) are imported from the legacy modules because they are
stable and identity-free. The small string-validation helpers below are private
to this module only because their legacy equivalents are private too, and this
slice may not edit legacy files to widen them.
"""

from __future__ import annotations

import re
import shlex
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .jsonio import read_json, read_jsonl, to_jsonable, write_json, write_jsonl
from .resources import ResourceSpec
from .specs import STAGE_MANIFEST, TASKS_JSONL, CompletionSpec
from .task_state import allocation_deadline_unix

#: Schema tag written into, and demanded of, every v2 stage manifest.
SCHEMA_VERSION_V2 = "experiment-toolkit/v2"

#: File name used by :func:`write_dispatch_records`.
DISPATCH_RECORDS_JSONL = "dispatch_records.jsonl"

#: Runtime key meaning "there is only one runtime and it has no distinguishing
#: name". Admission omits the runtime segment of ``attempt_id`` in that case; see
#: :func:`admit_tasks`.
DEFAULT_RUNTIME = "default"

# Context environment names are deliberately narrower than the process
# environment.  The executor owns the same literal independently so the
# experiment toolkit remains decoupled from the executor package.
CONTEXT_ENV_KEY_PATTERN = r"^(CUDA_VISIBLE_DEVICES|TPEN_CTX_[A-Z0-9_]{1,64})$"
_CONTEXT_ENV_KEY_RE = re.compile(CONTEXT_ENV_KEY_PATTERN)


@dataclass(frozen=True)
class LogicalTaskSpec:
    """One planned unit of science, independent of any attempt or runtime.

    Parameters
    ----------
    logical_task_id : str
        Planner-issued stable identity. **Opaque**: no consumer may parse it,
        re-derive it, or validate it against a convention.
    stage : str
        Stage this task belongs to.
    run_id : str
        Study-local run identity within the stage.
    command : tuple of str
        Science intent, not an executable command line. Tokens may contain
        placeholders such as ``{python}`` that admission resolves into a concrete
        ``argv``; this module never expands them.
    result_dir : str
        Directory that holds this task's results.
    inputs, outputs, logs : tuple of str
        Declared paths. ``logs[0]``, when present, is treated as the row status
        path by :func:`admit_tasks`.
    params : Mapping
        Free-form planner parameters.
    resources : ResourceSpec
        Resource request for the task.
    dependencies : tuple of str
        Logical ids (not attempt ids) this task depends on.
    completion : CompletionSpec
        Completion predicate. Completion is a *predicate over artefacts*, never
        an executor exit status.
    resume : Mapping
        Resume hints for the runner.
    metadata : Mapping
        Free-form provenance.
    """

    logical_task_id: str
    stage: str
    run_id: str
    command: tuple[str, ...]
    result_dir: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    logs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=lambda: ResourceSpec(profile="cpu", device="cpu"))
    dependencies: tuple[str, ...] = ()
    completion: CompletionSpec = field(default_factory=lambda: CompletionSpec(policy="none"))
    resume: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "LogicalTaskSpec":
        """Validate the logical-task contract and return ``self``.

        Validation is structural only. In particular ``logical_task_id`` is
        checked for non-emptiness and nothing else: it is never compared against
        :func:`logical_task_id_from_parts`.

        Returns
        -------
        LogicalTaskSpec
            ``self``, so validation can be chained onto construction.

        Raises
        ------
        ValueError
            If any required string is blank or any declared sequence contains a
            blank entry.
        """

        _require_non_empty("logical_task_id", self.logical_task_id)
        _require_non_empty("stage", self.stage)
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("result_dir", self.result_dir)
        if not self.command:
            raise ValueError(f"logical task {self.logical_task_id!r} command must be non-empty")
        _require_non_empty_sequence("command", self.command)
        _require_non_empty_sequence("inputs", self.inputs)
        _require_non_empty_sequence("outputs", self.outputs)
        _require_non_empty_sequence("logs", self.logs)
        _require_non_empty_sequence("dependencies", self.dependencies)
        self.resources.validate()
        self.completion.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns
        -------
        dict
            Serialized row. No attempt field is emitted, by design.
        """

        return {
            "logical_task_id": self.logical_task_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "command": list(self.command),
            "result_dir": self.result_dir,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "logs": list(self.logs),
            "params": to_jsonable(dict(self.params)),
            "resources": self.resources.to_dict(),
            "dependencies": list(self.dependencies),
            "completion": self.completion.to_dict(),
            "resume": to_jsonable(dict(self.resume)),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogicalTaskSpec":
        """Build a logical task spec from serialized data.

        Parameters
        ----------
        data : Mapping
            One row as produced by :meth:`to_dict`.

        Returns
        -------
        LogicalTaskSpec
            The validated spec.
        """

        return cls(
            logical_task_id=_required_str(data, "logical_task_id"),
            stage=_required_str(data, "stage"),
            run_id=_required_str(data, "run_id"),
            command=_string_tuple(data.get("command", ()), "command"),
            result_dir=_required_str(data, "result_dir"),
            inputs=_string_tuple(data.get("inputs", ()), "inputs"),
            outputs=_string_tuple(data.get("outputs", ()), "outputs"),
            logs=_string_tuple(data.get("logs", ()), "logs"),
            params=_mapping(data.get("params")),
            resources=ResourceSpec.from_dict(data.get("resources")),
            dependencies=_string_tuple(data.get("dependencies", ()), "dependencies"),
            completion=CompletionSpec.from_dict(data.get("completion")),
            resume=_mapping(data.get("resume")),
            metadata=_mapping(data.get("metadata")),
        ).validate()


def logical_task_id_from_parts(*, stage: str, run_id: str, plan_id: str) -> str:
    """Return the conventional logical task id for a planned row.

    This is a **convention helper for planners only**. Consumers must treat
    ``logical_task_id`` as opaque: nothing in this module, and nothing
    downstream, may call this function to re-derive or validate an existing id.
    The legacy surface did exactly that for ``task_id``, which welded attempt
    identity into task identity and made clean retries impossible.

    Parameters
    ----------
    stage : str
        Stage name.
    run_id : str
        Study-local run identity.
    plan_id : str
        Identity of the plan issuing the row.

    Returns
    -------
    str
        ``"{stage}:{run_id}:{plan_id}"``.

    Raises
    ------
    ValueError
        If any part is blank.
    """

    _require_non_empty("stage", stage)
    _require_non_empty("run_id", run_id)
    _require_non_empty("plan_id", plan_id)
    return f"{stage}:{run_id}:{plan_id}"


@dataclass(frozen=True)
class StagePlanV2:
    """Immutable, attempt-free stage-level task table.

    A plan describes intended science. It carries a ``plan_id`` for provenance
    but deliberately carries **no** attempt identity, so the same plan can be
    admitted any number of times.

    Parameters
    ----------
    study : str
        Study this plan belongs to.
    stage : str
        Stage name; every task row must agree with it.
    plan_id : str
        Provenance identity of this plan.
    results_root : str
        Root directory for stage results.
    tasks : tuple of LogicalTaskSpec
        Planned rows, with unique ``logical_task_id`` values.
    schema_version : str, default=``SCHEMA_VERSION_V2``
        Schema tag; the read path rejects anything else *before* parsing rows.
    """

    study: str
    stage: str
    plan_id: str
    results_root: str
    tasks: tuple[LogicalTaskSpec, ...]
    schema_version: str = SCHEMA_VERSION_V2

    @property
    def n_tasks(self) -> int:
        """Return the task count.

        Returns
        -------
        int
            Number of planned rows.
        """

        return len(self.tasks)

    def task_by_logical_id(self) -> dict[str, LogicalTaskSpec]:
        """Return the plan rows indexed by opaque logical id.

        Returns
        -------
        dict
            Mapping from ``logical_task_id`` to the planned row.
        """

        return {task.logical_task_id: task for task in self.tasks}

    def validate(self) -> "StagePlanV2":
        """Validate the plan contract and return ``self``.

        Returns
        -------
        StagePlanV2
            ``self``.

        Raises
        ------
        ValueError
            If a required string is blank, the schema tag is wrong, a row's stage
            disagrees with the plan, or a ``logical_task_id`` repeats.
        """

        _require_non_empty("study", self.study)
        _require_non_empty("stage", self.stage)
        _require_non_empty("plan_id", self.plan_id)
        _require_non_empty("results_root", self.results_root)
        if self.schema_version != SCHEMA_VERSION_V2:
            raise ValueError(
                f"stage plan schema_version {self.schema_version!r} does not match "
                f"{SCHEMA_VERSION_V2!r}"
            )
        seen: set[str] = set()
        for task in self.tasks:
            task.validate()
            if task.stage != self.stage:
                raise ValueError(
                    f"logical task {task.logical_task_id!r} stage {task.stage!r} does not match "
                    f"plan stage {self.stage!r}"
                )
            # Logical ids are opaque, so uniqueness is the *only* identity
            # invariant a plan can assert about them.
            if task.logical_task_id in seen:
                raise ValueError(
                    f"duplicate logical_task_id in stage plan: {task.logical_task_id!r}"
                )
            seen.add(task.logical_task_id)
        return self

    def to_manifest(self) -> dict[str, Any]:
        """Return the compact stage manifest.

        Returns
        -------
        dict
            Manifest payload written to ``stage_manifest.json``.
        """

        return {
            # schema_version leads the payload so a reader can reject an
            # incompatible plan by inspecting the first key it meets.
            "schema_version": self.schema_version,
            "study": self.study,
            "stage": self.stage,
            "plan_id": self.plan_id,
            "results_root": self.results_root,
            "tasks_path": TASKS_JSONL,
            "n_tasks": self.n_tasks,
        }

    def write(self, directory: str | Path) -> Path:
        """Write the manifest and task rows into ``directory``.

        Parameters
        ----------
        directory : str or Path
            Destination directory; created if missing.

        Returns
        -------
        Path
            The destination directory.
        """

        self.validate()
        directory = Path(directory)
        write_json(directory / STAGE_MANIFEST, self.to_manifest())
        write_jsonl(directory / TASKS_JSONL, (task.to_dict() for task in self.tasks))
        return directory

    @classmethod
    def read(cls, directory: str | Path) -> "StagePlanV2":
        """Read a v2 stage plan from ``directory``.

        The schema tag is checked **before any task row is parsed**, so a plan
        written by a different schema fails with a schema error rather than with
        whatever incidental parse error its rows happen to raise.

        Parameters
        ----------
        directory : str or Path
            Directory holding ``stage_manifest.json`` and the task rows.

        Returns
        -------
        StagePlanV2
            The validated plan.

        Raises
        ------
        ValueError
            If the schema tag is absent or not ``SCHEMA_VERSION_V2``, or if the
            row count disagrees with the manifest.
        """

        directory = Path(directory)
        manifest = read_json(directory / STAGE_MANIFEST)
        schema_version = _required_str(manifest, "schema_version")
        if schema_version != SCHEMA_VERSION_V2:
            raise ValueError(
                f"stage manifest schema_version {schema_version!r} is not {SCHEMA_VERSION_V2!r}; "
                "refusing to parse task rows"
            )
        # Only now is it safe to interpret rows under v2 field expectations.
        tasks_path = _required_str(manifest, "tasks_path")
        tasks = tuple(LogicalTaskSpec.from_dict(row) for row in read_jsonl(directory / tasks_path))
        expected_n_tasks = _required_int(manifest, "n_tasks")
        plan = cls(
            study=_required_str(manifest, "study"),
            stage=_required_str(manifest, "stage"),
            plan_id=_required_str(manifest, "plan_id"),
            results_root=_required_str(manifest, "results_root"),
            schema_version=schema_version,
            tasks=tasks,
        )
        if expected_n_tasks != plan.n_tasks:
            raise ValueError(
                f"stage manifest n_tasks={expected_n_tasks} does not match {plan.n_tasks} task rows"
            )
        return plan.validate()


@dataclass(frozen=True)
class DispatchSpec:
    """One admitted attempt at one logical task on one runtime.

    A dispatch is fully resolved: ``argv`` is the exact, immutable command an
    executor must submit unchanged. Validation is intentionally limited to
    non-empty checks — in particular ``attempt_id`` is **not** re-derived from
    ``admission_id`` and ``logical_task_id``, because it is opaque downstream.

    Parameters
    ----------
    logical_task_id : str
        Opaque planner identity this attempt targets.
    admission_id : str
        Identity of the admission that minted this attempt. This is the claim
        scope: a retry uses a fresh ``admission_id``.
    attempt_id : str
        Opaque attempt identity minted by :func:`admit_tasks`.
    stage, run_id : str
        Copied from the planned row for routing and reporting.
    argv : tuple of str
        Exact command to submit. Executors must not mutate it.
    result_dir : str
        Directory that holds this task's results.
    runtime : str
        Logical runtime reference, e.g. ``"tpen-cu126"``. It names an
        environment; it is not a path and is not resolved here.
    cwd : str
        Working directory for the submitted command.
    environment : Mapping of str to str
        Extra environment entries for the submitted command.
    completion : CompletionSpec
        Artefact predicate deciding whether the work is done.
    row_status_path : str, optional
        Path of the per-row status file, when the planner declared one.
    metadata : Mapping
        Provenance only. ``metadata["admission_ref"]`` records a caller-supplied
        reference and carries no identity meaning.
    """

    logical_task_id: str
    admission_id: str
    attempt_id: str
    stage: str
    run_id: str
    argv: tuple[str, ...]
    result_dir: str
    runtime: str
    cwd: str
    environment: Mapping[str, str] = field(default_factory=dict)
    completion: CompletionSpec = field(default_factory=lambda: CompletionSpec(policy="none"))
    row_status_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "DispatchSpec":
        """Validate the dispatch contract and return ``self``.

        Returns
        -------
        DispatchSpec
            ``self``.

        Raises
        ------
        ValueError
            If any required string is blank or ``argv`` is empty or contains a
            blank token.
        """

        _require_non_empty("logical_task_id", self.logical_task_id)
        _require_non_empty("admission_id", self.admission_id)
        _require_non_empty("attempt_id", self.attempt_id)
        _require_non_empty("stage", self.stage)
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("result_dir", self.result_dir)
        _require_non_empty("runtime", self.runtime)
        _require_non_empty("cwd", self.cwd)
        if not self.argv:
            raise ValueError(f"dispatch {self.attempt_id!r} argv must be non-empty")
        _require_non_empty_sequence("argv", self.argv)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns
        -------
        dict
            Serialized dispatch.
        """

        return {
            "logical_task_id": self.logical_task_id,
            "admission_id": self.admission_id,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "argv": list(self.argv),
            "argv_text": shlex.join(self.argv),
            "result_dir": self.result_dir,
            "runtime": self.runtime,
            "cwd": self.cwd,
            "environment": {str(key): str(value) for key, value in self.environment.items()},
            "completion": self.completion.to_dict(),
            "row_status_path": self.row_status_path,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DispatchSpec":
        """Build a dispatch spec from serialized data.

        Parameters
        ----------
        data : Mapping
            Payload as produced by :meth:`to_dict`.

        Returns
        -------
        DispatchSpec
            The validated dispatch.
        """

        return cls(
            logical_task_id=_required_str(data, "logical_task_id"),
            admission_id=_required_str(data, "admission_id"),
            attempt_id=_required_str(data, "attempt_id"),
            stage=_required_str(data, "stage"),
            run_id=_required_str(data, "run_id"),
            argv=_string_tuple(data.get("argv", ()), "argv"),
            result_dir=_required_str(data, "result_dir"),
            runtime=_required_str(data, "runtime"),
            cwd=_required_str(data, "cwd"),
            environment={
                str(key): str(value) for key, value in _mapping(data.get("environment")).items()
            },
            completion=CompletionSpec.from_dict(data.get("completion")),
            row_status_path=_optional_str(data.get("row_status_path")),
            metadata=_mapping(data.get("metadata")),
        ).validate()


@dataclass(frozen=True)
class AllocationContext:
    """Allocation attach input, carried as data rather than executor state.

    An executor that attaches to an existing allocation needs the allocation's
    identity, its per-worker accelerator visibility binding, and when it must
    stop claiming new work. Passing that as a value keeps executors stateless and
    makes attach decisions reproducible from a record.

    Parameters
    ----------
    allocation_id : str
        Identity of the allocation being attached to.
    visibility_variable : str
        Per-worker environment variable, such as ``CUDA_VISIBLE_DEVICES``.
    visibility_values : tuple of str
        One visibility value per worker, in worker-index order. An empty tuple
        means inherit mode: no per-worker visibility binding is applied because
        the scheduler owns the variable.
    run_root : str, optional
        Root under which claim state lives. ``None`` means the caller decides.
    deadline : str or float, optional
        Explicit allocation deadline, as UNIX seconds or an ISO timestamp.
    deadline_env_var : str, optional
        Facility-provided deadline variable consulted before the Slurm fallback.
    deadline_guard_min : int, default=1
        Minutes before the deadline at which workers stop claiming new work.
    environment : Mapping of str to str
        Extra environment entries inherited by every task.
    nodes_per_block : int, optional
        Number of nodes required per execution block. ``None`` preserves the
        legacy single-node attach behavior and is omitted from serialization.
    """

    allocation_id: str
    visibility_variable: str
    visibility_values: tuple[str, ...]
    run_root: str | None = None
    deadline: str | float | None = None
    deadline_env_var: str | None = None
    deadline_guard_min: int = 1
    environment: Mapping[str, str] = field(default_factory=dict)
    nodes_per_block: int | None = None

    def validate(self) -> "AllocationContext":
        """Validate the attach contract and return ``self``.

        Returns
        -------
        AllocationContext
            ``self``.

        Raises
        ------
        ValueError
            If a required string is blank, ``visibility_values`` contains a
            blank entry, or the guard is negative. An empty
            ``visibility_values`` tuple is valid inherit mode.
        """

        _require_non_empty("allocation_id", self.allocation_id)
        if type(self.visibility_variable) is not str or not _CONTEXT_ENV_KEY_RE.fullmatch(
            self.visibility_variable
        ):
            raise ValueError(
                "context-environment-key: allocation context visibility_variable must match "
                f"{CONTEXT_ENV_KEY_PATTERN!r}"
            )
        _require_non_empty_sequence("visibility_values", self.visibility_values)
        if not isinstance(self.environment, Mapping):
            raise ValueError("context-environment-key: allocation context environment must be a mapping")
        invalid_keys = sorted(
            key
            for key in self.environment
            if type(key) is not str or _CONTEXT_ENV_KEY_RE.fullmatch(key) is None
        )
        invalid_keys.sort(key=repr)
        if invalid_keys:
            raise ValueError(
                "context-environment-key: allocation context declares unsupported keys "
                f"{invalid_keys!r}"
            )
        if any(type(value) is not str for value in self.environment.values()):
            raise ValueError("context-environment-key: allocation context values must be strings")
        if self.deadline_guard_min < 0:
            raise ValueError("allocation context deadline_guard_min must be non-negative")
        if self.nodes_per_block is not None and self.nodes_per_block <= 0:
            raise ValueError("allocation context nodes_per_block must be positive")
        return self

    def deadline_unix(self, environ: Mapping[str, str] | None = None) -> float | None:
        """Resolve the allocation deadline through the shared toolkit helper.

        Parameters
        ----------
        environ : Mapping, optional
            Environment mapping to consult. Defaults to the process environment.

        Returns
        -------
        float or None
            UNIX deadline in seconds, or ``None`` when no source supplied one.
        """

        # Reuse the legacy resolution order (explicit, then named variable, then
        # SLURM_JOB_END_TIME) rather than re-implementing facility precedence.
        return allocation_deadline_unix(
            self.deadline,
            env_var=self.deadline_env_var,
            environ=environ,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns
        -------
        dict
            Serialized attach context.
        """

        serialized = {
            "allocation_id": self.allocation_id,
            "visibility_variable": self.visibility_variable,
            "visibility_values": list(self.visibility_values),
            "run_root": self.run_root,
            "deadline": self.deadline,
            "deadline_env_var": self.deadline_env_var,
            "deadline_guard_min": self.deadline_guard_min,
            "environment": {str(key): str(value) for key, value in self.environment.items()},
        }
        if self.nodes_per_block is not None:
            serialized["nodes_per_block"] = self.nodes_per_block
        return serialized


def mint_admission_id(label: str) -> str:
    """Mint a fresh admission identity.

    Every call returns a distinct value. That is the point: an ``admission_id``
    *is* the retry boundary, so re-admitting the same tasks under a new admission
    is how a retry is expressed.

    Parameters
    ----------
    label : str
        Human-readable prefix, e.g. ``"he-cutover-smoke"``. Characters outside
        ``[A-Za-z0-9._-]`` are replaced with ``-`` so the id is safe in file
        names and environment values.

    Returns
    -------
    str
        ``"{label}-{UTC timestamp}-{random suffix}"``.

    Raises
    ------
    ValueError
        If ``label`` is blank.

    Notes
    -----
    The timestamp is UTC even though experiment *logging* uses
    ``America/New_York``: an identifier must be unambiguous and monotonically
    sortable across facilities, and must not shift under a daylight-saving
    transition.
    """

    _require_non_empty("label", label)
    safe_label = "".join(
        character if (character.isalnum() or character in "._-") else "-" for character in label
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_label}-{stamp}-{uuid.uuid4().hex[:8]}"


def admit_tasks(
    plan: StagePlanV2,
    tasks: Sequence[LogicalTaskSpec],
    *,
    admission_id: str,
    argv_by_runtime: Mapping[str, Sequence[Sequence[str]]],
    cwd: str | Path,
    environment: Mapping[str, str],
    admission_ref: str | None = None,
) -> tuple[DispatchSpec, ...]:
    """Admit a subset of a plan into concrete dispatches.

    This is the only place attempts are minted. One :class:`DispatchSpec` is
    produced per (task x runtime) pair, in task order then runtime order.

    Attempt identity is deterministic given its inputs::

        attempt_id = f"{admission_id}:{logical_task_id}"            # single unnamed runtime
        attempt_id = f"{admission_id}:{logical_task_id}:{runtime}"  # otherwise

    "Single unnamed runtime" means ``argv_by_runtime`` has exactly one key and
    that key is :data:`DEFAULT_RUNTIME`: the runtime carries no distinguishing
    name, so appending it would add no information. The result is opaque
    downstream — consumers must never parse it back apart.

    Parameters
    ----------
    plan : StagePlanV2
        Plan the admitted tasks must come from.
    tasks : Sequence of LogicalTaskSpec
        Subset of ``plan.tasks`` to admit. Order is preserved.
    admission_id : str
        Claim scope for this admission, typically from :func:`mint_admission_id`.
    argv_by_runtime : Mapping of str to Sequence of Sequence of str
        For each runtime, one exact ``argv`` per entry of ``tasks``, in the same
        order.
    cwd : str or Path
        Working directory for every submitted command.
    environment : Mapping of str to str
        Extra environment entries for every submitted command.
    admission_ref : str, optional
        Free-form provenance reference recorded under
        ``metadata["admission_ref"]``. It has **no** identity meaning: two
        admissions sharing a ref still mint different attempts.

    Returns
    -------
    tuple of DispatchSpec
        The minted dispatches.

    Raises
    ------
    ValueError
        If ``admission_id`` is blank, ``tasks`` is empty, a task is not in the
        plan, a ``logical_task_id`` repeats within ``tasks``,
        ``argv_by_runtime`` is empty, a runtime name is blank, or a runtime's
        argv list length does not match ``len(tasks)``.
    """

    _require_non_empty("admission_id", admission_id)
    if not tasks:
        raise ValueError("admit_tasks requires at least one task")
    if not argv_by_runtime:
        raise ValueError("admit_tasks requires at least one runtime in argv_by_runtime")

    plan.validate()
    planned = plan.task_by_logical_id()

    # Subset and duplicate checks: admission may only admit planned rows, and may
    # not admit the same row twice within one admission.
    seen: set[str] = set()
    for task in tasks:
        task.validate()
        if task.logical_task_id not in planned:
            raise ValueError(
                f"logical_task_id {task.logical_task_id!r} is not in stage plan "
                f"{plan.plan_id!r}"
            )
        if task != planned[task.logical_task_id]:
            raise ValueError(
                f"logical task {task.logical_task_id!r} does not match its stage-plan row"
            )
        if task.logical_task_id in seen:
            raise ValueError(
                f"duplicate logical_task_id in admitted tasks: {task.logical_task_id!r}"
            )
        seen.add(task.logical_task_id)

    # Alignment check: every runtime must supply exactly one argv per task.
    for runtime, argv_rows in argv_by_runtime.items():
        _require_non_empty("runtime", runtime)
        if isinstance(argv_rows, str):
            raise ValueError(f"argv_by_runtime[{runtime!r}] must be a sequence of argv, not a string")
        if len(argv_rows) != len(tasks):
            raise ValueError(
                f"argv_by_runtime[{runtime!r}] length {len(argv_rows)} does not match "
                f"tasks length {len(tasks)}"
            )

    runtimes = list(argv_by_runtime)
    # A lone DEFAULT_RUNTIME adds nothing to the attempt identity, so it is
    # omitted; any named runtime, or more than one runtime, must appear.
    include_runtime_segment = not (len(runtimes) == 1 and runtimes[0] == DEFAULT_RUNTIME)

    dispatches: list[DispatchSpec] = []
    for task_index, task in enumerate(tasks):
        for runtime in runtimes:
            attempt_id = (
                f"{admission_id}:{task.logical_task_id}:{runtime}"
                if include_runtime_segment
                else f"{admission_id}:{task.logical_task_id}"
            )
            dispatches.append(
                DispatchSpec(
                    logical_task_id=task.logical_task_id,
                    admission_id=admission_id,
                    attempt_id=attempt_id,
                    stage=task.stage,
                    run_id=task.run_id,
                    argv=tuple(str(part) for part in argv_by_runtime[runtime][task_index]),
                    result_dir=task.result_dir,
                    runtime=runtime,
                    cwd=str(cwd),
                    environment={str(key): str(value) for key, value in environment.items()},
                    completion=task.completion,
                    # The planner declares the row status file first in `logs`;
                    # admission copies it rather than inventing a path.
                    row_status_path=task.logs[0] if task.logs else None,
                    metadata={
                        "study": plan.study,
                        "plan_id": plan.plan_id,
                        # Provenance only; never an identity input.
                        "admission_ref": admission_ref,
                    },
                ).validate()
            )
    return tuple(dispatches)


@dataclass(frozen=True)
class DispatchRecord:
    """Receipt that a backend *accepted* a dispatch.

    A record attests acceptance, not success. Whether the science finished is
    decided by the :class:`~experiments.toolkit.specs.CompletionSpec` predicate
    over artefacts, never by an exit status recorded here.

    Parameters
    ----------
    logical_task_id, admission_id, attempt_id, stage, run_id : str
        Identity copied verbatim from the accepted :class:`DispatchSpec`.
    backend : str
        Backend that accepted the dispatch, e.g. ``"parsl"``.
    launcher_job_id : str
        Backend-assigned job identity.
    submitted_command : tuple of str
        Command actually submitted. It must equal the dispatch's ``argv``; see
        :meth:`accepted`.
    runtime : str
        Logical runtime reference the dispatch was accepted for.
    status_path : str, optional
        Row status path carried over from the dispatch.
    metadata : Mapping
        Free-form provenance.
    """

    logical_task_id: str
    admission_id: str
    attempt_id: str
    stage: str
    run_id: str
    backend: str
    launcher_job_id: str
    submitted_command: tuple[str, ...]
    runtime: str
    status_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "DispatchRecord":
        """Validate the record contract and return ``self``.

        Returns
        -------
        DispatchRecord
            ``self``.

        Raises
        ------
        ValueError
            If a required string is blank or ``submitted_command`` is empty or
            contains a blank token.
        """

        _require_non_empty("logical_task_id", self.logical_task_id)
        _require_non_empty("admission_id", self.admission_id)
        _require_non_empty("attempt_id", self.attempt_id)
        _require_non_empty("stage", self.stage)
        _require_non_empty("run_id", self.run_id)
        _require_non_empty("backend", self.backend)
        _require_non_empty("launcher_job_id", self.launcher_job_id)
        _require_non_empty("runtime", self.runtime)
        if not self.submitted_command:
            raise ValueError(f"dispatch record {self.attempt_id!r} submitted_command must be non-empty")
        _require_non_empty_sequence("submitted_command", self.submitted_command)
        return self

    @classmethod
    def accepted(
        cls,
        dispatch: DispatchSpec,
        *,
        backend: str,
        launcher_job_id: str,
        submitted_command: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> "DispatchRecord":
        """Build a record for an accepted dispatch.

        Parameters
        ----------
        dispatch : DispatchSpec
            The dispatch the backend accepted.
        backend : str
            Backend name.
        launcher_job_id : str
            Backend-assigned job identity.
        submitted_command : Sequence of str
            What the backend actually submitted. It must equal ``dispatch.argv``
            exactly: executors are forbidden from mutating ``argv``, and this is
            where that rule is enforced rather than merely documented.
        metadata : Mapping, optional
            Extra provenance.

        Returns
        -------
        DispatchRecord
            The validated record.

        Raises
        ------
        ValueError
            If ``submitted_command`` differs from ``dispatch.argv``.
        """

        submitted = tuple(str(part) for part in submitted_command)
        if submitted != dispatch.argv:
            raise ValueError(
                f"dispatch record for {dispatch.attempt_id!r} submitted_command "
                f"{submitted!r} does not equal dispatch argv {dispatch.argv!r}"
            )
        return cls(
            logical_task_id=dispatch.logical_task_id,
            admission_id=dispatch.admission_id,
            attempt_id=dispatch.attempt_id,
            stage=dispatch.stage,
            run_id=dispatch.run_id,
            backend=backend,
            launcher_job_id=str(launcher_job_id),
            submitted_command=submitted,
            runtime=dispatch.runtime,
            status_path=dispatch.row_status_path,
            metadata=dict(metadata or {}),
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping.

        Returns
        -------
        dict
            Serialized record.
        """

        return {
            "logical_task_id": self.logical_task_id,
            "admission_id": self.admission_id,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "backend": self.backend,
            "launcher_job_id": self.launcher_job_id,
            "submitted_command": list(self.submitted_command),
            "submitted_command_text": shlex.join(self.submitted_command),
            "runtime": self.runtime,
            "status_path": self.status_path,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DispatchRecord":
        """Build a record from serialized data.

        Parameters
        ----------
        data : Mapping
            Payload as produced by :meth:`to_dict`.

        Returns
        -------
        DispatchRecord
            The validated record.
        """

        return cls(
            logical_task_id=_required_str(data, "logical_task_id"),
            admission_id=_required_str(data, "admission_id"),
            attempt_id=_required_str(data, "attempt_id"),
            stage=_required_str(data, "stage"),
            run_id=_required_str(data, "run_id"),
            backend=_required_str(data, "backend"),
            launcher_job_id=_required_str(data, "launcher_job_id"),
            submitted_command=_string_tuple(data.get("submitted_command", ()), "submitted_command"),
            runtime=_required_str(data, "runtime"),
            status_path=_optional_str(data.get("status_path")),
            metadata=_mapping(data.get("metadata")),
        ).validate()


def write_dispatch_records(directory: str | Path, records: Sequence[DispatchRecord]) -> Path:
    """Write dispatch records next to a stage plan.

    Parameters
    ----------
    directory : str or Path
        Destination directory; created if missing.
    records : Sequence of DispatchRecord
        Records to write, one JSON object per line.

    Returns
    -------
    Path
        Path of the written ``dispatch_records.jsonl``.
    """

    path = Path(directory) / DISPATCH_RECORDS_JSONL
    write_jsonl(path, (record.validate().to_dict() for record in records))
    return path


def read_dispatch_records(directory: str | Path) -> tuple[DispatchRecord, ...]:
    """Read dispatch records written by :func:`write_dispatch_records`.

    Parameters
    ----------
    directory : str or Path
        Directory holding ``dispatch_records.jsonl``.

    Returns
    -------
    tuple of DispatchRecord
        The validated records, in file order.
    """

    path = Path(directory) / DISPATCH_RECORDS_JSONL
    return tuple(DispatchRecord.from_dict(row) for row in read_jsonl(path))


@runtime_checkable
class DispatchExecutor(Protocol):
    """Backend that submits ready-made dispatches and reports acceptance.

    Implementations are deliberately narrow. An executor:

    - **never plans** — it receives dispatches; it does not select, expand, or
      order work;
    - **never retries** — a retry is a fresh ``admission_id`` from
      :func:`admit_tasks`, not a re-submission decided inside a backend;
    - **never mutates argv** — it submits :attr:`DispatchSpec.argv` verbatim, and
      :meth:`DispatchRecord.accepted` rejects any deviation;
    - **never infers completion from exit status** — completion is the
      :class:`~experiments.toolkit.specs.CompletionSpec` predicate over
      artefacts; an exit code says only that a process stopped.
    """

    def dispatch(
        self,
        dispatches: Sequence[DispatchSpec],
        *,
        context: AllocationContext | None,
    ) -> Sequence[DispatchRecord]:
        """Submit ``dispatches`` and return one acceptance record per dispatch.

        Parameters
        ----------
        dispatches : Sequence of DispatchSpec
            Fully resolved attempts to submit.
        context : AllocationContext or None
            Attach input when submitting into an existing allocation; ``None``
            when the backend provisions its own resources.

        Returns
        -------
        Sequence of DispatchRecord
            Acceptance receipts, one per dispatch.
        """
        ...


def _optional_str(value: Any) -> str | None:
    """Return ``value`` as a string, or ``None`` when absent or blank."""

    if value is None or value == "":
        return None
    return str(value)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    """Return a required non-blank string field from ``data``."""

    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing required field: {key}")
    return str(value)


def _required_int(data: Mapping[str, Any], key: str) -> int:
    """Return a required integer field from ``data``."""

    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return int(value)


def _require_non_empty(name: str, value: str) -> None:
    """Raise when ``value`` is blank."""

    if not str(value).strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_empty_sequence(name: str, values: Sequence[str]) -> None:
    """Raise when any entry of ``values`` is blank, naming the offending indexes."""

    empty = [index for index, value in enumerate(values) if not str(value).strip()]
    if empty:
        raise ValueError(f"{name} contains empty entries at indexes: {empty}")


def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` as a plain dict, or an empty dict when it is not a mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of strings, rejecting bare strings."""

    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence, not a string")
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
