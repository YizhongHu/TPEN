"""Pure planning for the baseline comparison matrix.

This module only describes work.  Admission resolves the injected command for
an execution environment; this builder never invokes a shell, scheduler, or
serialization format for executable arguments.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.toolkit.dispatch import (
    LogicalTaskSpec,
    StagePlanV2,
    logical_task_id_from_parts,
)
from experiments.toolkit.executor_inputs import ready_group
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec

MATRIX_FIELDS = ("code", "ansatz", "system", "seed", "steps", "batch_size")


def build_plan(
    matrix: Mapping[str, Sequence[Any]],
    *,
    command: tuple[str, ...],
    results_root: str | Path,
    plan_id: str,
    study: str = "baselines",
    stage: str = "baselines",
    resources: ResourceSpec | None = None,
) -> StagePlanV2:
    """Build a deterministic, attempt-free plan for every matrix row.

    Parameters
    ----------
    matrix : Mapping[str, Sequence]
        The six-dimensional declarative matrix.  Every dimension must be
        present and contain unique, non-``None`` values.  An empty dimension
        deliberately produces an empty plan.
    command : tuple of str
        Exact argv supplied by the caller.  Command construction is owned by
        the baseline adapter and is intentionally not performed here.
    results_root : str or pathlib.Path
        Root under which each row receives its own result directory.
    plan_id : str
        Stable identity for this plan.
    study, stage : str, optional
        Toolkit routing labels.
    resources : ResourceSpec, optional
        GPU resource request.  The default is one CUDA GPU.

    Returns
    -------
    StagePlanV2
        A validated plan containing one logical task per Cartesian-product row.

    Raises
    ------
    ValueError
        If the matrix shape, command, or resource request is invalid.
    """

    dimensions = _validate_matrix(matrix)
    argv = _validate_command(command)
    gpu_resources = resources or ResourceSpec(profile="cuda", device="cuda", gpus=1)
    gpu_resources.validate()
    if gpu_resources.device != "cuda" or not gpu_resources.gpus or gpu_resources.gpus < 1:
        raise ValueError("baseline tasks require a CUDA resource with at least one GPU")

    root = str(results_root)
    if not root.strip():
        raise ValueError("results_root must be a non-empty string")

    tasks = tuple(
        _task_for_row(
            row,
            command=argv,
            results_root=root,
            plan_id=plan_id,
            stage=stage,
            resources=gpu_resources,
        )
        for row in itertools.product(*(dimensions[field] for field in MATRIX_FIELDS))
    )
    return StagePlanV2(
        study=study,
        stage=stage,
        plan_id=plan_id,
        results_root=root,
        tasks=tasks,
    ).validate()


def executor_input(plan: StagePlanV2) -> dict[str, object]:
    """Emit the executor input mapping for a built plan.

    Parameters
    ----------
    plan : StagePlanV2
        Validated baseline plan to describe.

    Returns
    -------
    dict
        Plain executor ``ReadyGroup`` wire mapping.
    """

    return ready_group(plan)


def _validate_matrix(matrix: Mapping[str, Sequence[Any]]) -> dict[str, tuple[Any, ...]]:
    if not isinstance(matrix, Mapping):
        raise ValueError("matrix must be a mapping of dimension names to sequences")
    missing = [field for field in MATRIX_FIELDS if field not in matrix]
    if missing:
        raise ValueError(f"matrix is missing dimensions: {missing}")
    dimensions: dict[str, tuple[Any, ...]] = {}
    for field in MATRIX_FIELDS:
        values = matrix[field]
        if values is None or isinstance(values, (str, bytes)):
            raise ValueError(f"matrix dimension {field!r} must be a sequence")
        try:
            values_tuple = tuple(values)
        except TypeError as exc:
            raise ValueError(f"matrix dimension {field!r} must be a sequence") from exc
        if any(value is None for value in values_tuple):
            raise ValueError(f"matrix dimension {field!r} cannot contain None")
        if len({_stable_value(value) for value in values_tuple}) != len(values_tuple):
            raise ValueError(f"matrix dimension {field!r} contains duplicate values")
        dimensions[field] = values_tuple
    return dimensions


def _validate_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(command, tuple) or not command:
        raise ValueError("command must be a non-empty argv tuple")
    if any(not isinstance(token, str) or not token.strip() for token in command):
        raise ValueError("command must contain non-empty string tokens")
    return command


def _task_for_row(
    row: tuple[Any, ...],
    *,
    command: tuple[str, ...],
    results_root: str,
    plan_id: str,
    stage: str,
    resources: ResourceSpec,
) -> LogicalTaskSpec:
    values = dict(zip(MATRIX_FIELDS, row))
    row_key = _canonical_row(values)
    row_digest = hashlib.sha256(row_key.encode("utf-8")).hexdigest()[:12]
    run_id = f"row-{row_digest}"
    logical_id = logical_task_id_from_parts(stage=stage, run_id=run_id, plan_id=plan_id)
    result_dir = str(Path(results_root) / _row_directory(values, row_digest))
    status_path = str(Path(result_dir) / "status.json")
    return LogicalTaskSpec(
        logical_task_id=logical_id,
        stage=stage,
        run_id=run_id,
        command=command,
        result_dir=result_dir,
        outputs=(status_path,),
        logs=(status_path,),
        params=values,
        resources=resources,
        completion=CompletionSpec(policy="status_completed", status_path=status_path),
        metadata={"matrix": values, "row_key": row_key},
    )


def _canonical_row(values: Mapping[str, Any]) -> str:
    return json.dumps({field: values[field] for field in MATRIX_FIELDS}, sort_keys=True, separators=(",", ":"), default=str)


def _row_directory(values: Mapping[str, Any], digest: str) -> str:
    readable = "__".join(f"{field}-{_safe_token(values[field])}" for field in MATRIX_FIELDS)
    return f"{readable}__{digest}"


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return token or "value"


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


__all__ = ["MATRIX_FIELDS", "build_plan"]
