"""Explicit opt-in He-v1 TRAIN activation through the attached executor.

The normal He-v1 launcher remains unchanged.  This module is a bounded route
for one declared TRAIN generation: it translates row identity and overrides,
optionally derives the smoke-only reduced callback config, checks the declared
core profile, writes the two executor handoff files, and invokes the public
executor CLI.  It does not implement evaluation, baselines, retries, or
campaign scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.toolkit.attached_activation import (
    ActivationRefusal,
    AttachedHandoff,
    ExecutorDispatchResult,
    PROFILE_NAME,
    ProfileVerdict,
    bind_allocation_context,
    check_profile,
    file_sha256,
    retire_attached_generation,
    run_executor_cli,
    write_activation_failure_receipt,
    write_activation_receipt,
    write_attached_handoff,
)
from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2


REDUCED_CONFIG_SCHEMA = "tpen-experiments/he-train-reduced-config-v1"
ROUTE_NAME = "he-train-attached"
HE_TRAIN_STAGE = "02_train"
ALLOWED_REDUCED_CALLBACKS = frozenset(
    {
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.Status",
        "tpen.callback.Checkpoint",
    }
)
ALLOWED_REDUCED_LOGGERS = frozenset({"tpen.logging.CSV"})


@dataclass(frozen=True, slots=True)
class ReducedConfig:
    """A smoke-only effective config and its durable comparison artifacts."""

    source_path: str
    effective_path: str
    diff_path: str
    hash_path: str
    source_sha256: str
    effective_sha256: str
    checkpoint_every_n: int


@dataclass(frozen=True, slots=True)
class PreparedHeTrain:
    """Prepared He TRAIN route inputs before profile/CLI execution."""

    plan: StagePlanV2
    handoff: AttachedHandoff
    effective_config: str
    source_config: str
    checkpoint_every_n: int


@dataclass(frozen=True, slots=True)
class HeTrainActivation:
    """The operational CLI result plus the root-reader receipt path."""

    prepared: PreparedHeTrain
    profile: ProfileVerdict
    dispatch: ExecutorDispatchResult
    receipt_path: str


def _absolute(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ActivationRefusal("config-path", f"{name} must be absolute")
    return path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _target(entry: Mapping[str, Any], name: str) -> str:
    if not isinstance(entry, Mapping):
        raise ActivationRefusal("config-shape", f"{name} must be a mapping")
    target = entry.get("_target_")
    if type(target) is not str or not target:
        raise ActivationRefusal("config-shape", f"{name} has no explicit _target_")
    return target


def _checkpoint_entry(callbacks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    entries = [entry for entry in callbacks if _target(entry, "callback") == "tpen.callback.Checkpoint"]
    if len(entries) != 1:
        raise ActivationRefusal("config-shape", "He TRAIN config must contain exactly one Checkpoint callback")
    return entries[0]


def _checkpoint_cadence(callback: Mapping[str, Any]) -> int:
    schedule = callback.get("schedule")
    if not isinstance(schedule, Mapping) or type(schedule.get("every_n")) is not int or schedule["every_n"] <= 0:
        raise ActivationRefusal("config-shape", "Checkpoint.schedule.every_n must be one positive integer")
    return int(schedule["every_n"])


def generate_reduced_callback_config(
    source_path: str | Path,
    output_path: str | Path,
    *,
    enable_reduced_callbacks: bool = False,
    checkpoint_every_n: int | None = None,
) -> ReducedConfig:
    """Derive the disclosed reduced callback config for a smoke only.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Absolute legacy He YAML read without modification.
    output_path : str or pathlib.Path
        Distinct absolute path for the effective smoke YAML.
    enable_reduced_callbacks : bool, default=False
        Explicit opt-in guard.  Without it the function refuses rather than
        silently changing the legacy He config.
    checkpoint_every_n : int, optional
        One positive cadence to write into the unique Checkpoint callback.

    Returns
    -------
    ReducedConfig
        Effective config, diff, hashes, and source provenance.

    Raises
    ------
    ActivationRefusal
        If opt-in, config shape, callback policy, cadence, or source/output
        provenance is invalid.
    """

    source = _absolute(source_path, "source_config")
    output = _absolute(output_path, "effective_config")
    if not enable_reduced_callbacks:
        raise ActivationRefusal("reduced-config-opt-in", "reduced callbacks require explicit smoke opt-in")
    if source.is_symlink() or output.is_symlink() or source.resolve() == output.resolve():
        raise ActivationRefusal(
            "config-alias",
            "source_config and effective_config must be distinct non-symlink paths",
        )
    try:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        document = yaml.safe_load(source_text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ActivationRefusal("config-read", f"source config could not be loaded: {error}") from error
    if not isinstance(document, dict):
        raise ActivationRefusal("config-shape", "source config must be a mapping")
    callbacks = document.get("callbacks")
    loggers = document.get("loggers")
    if not isinstance(callbacks, list) or not isinstance(loggers, list):
        raise ActivationRefusal("config-shape", "He TRAIN config callbacks and loggers must be lists")
    original_science = {key: document.get(key) for key in ("physics", "model", "optimizer", "sampler", "system")}
    original_callbacks = [_target(entry, "callback") for entry in callbacks if isinstance(entry, Mapping)]
    if len(original_callbacks) != len(callbacks):
        raise ActivationRefusal("config-shape", "every callback must be a mapping")
    checkpoint = _checkpoint_entry(callbacks)
    cadence = _checkpoint_cadence(checkpoint) if checkpoint_every_n is None else checkpoint_every_n
    if type(cadence) is not int or cadence <= 0:
        raise ActivationRefusal("config-shape", "checkpoint_every_n must be a positive integer")
    reduced = [entry for entry in callbacks if _target(entry, "callback") in ALLOWED_REDUCED_CALLBACKS]
    if not reduced or _target(_checkpoint_entry(reduced), "callback") != "tpen.callback.Checkpoint":
        raise ActivationRefusal("config-shape", "reduced config lost the required Checkpoint callback")
    reduced_checkpoint = dict(_checkpoint_entry(reduced))
    schedule = dict(reduced_checkpoint["schedule"])
    schedule["every_n"] = cadence
    reduced_checkpoint["schedule"] = schedule
    reduced[reduced.index(_checkpoint_entry(reduced))] = reduced_checkpoint
    reduced_loggers = [entry for entry in loggers if _target(entry, "logger") in ALLOWED_REDUCED_LOGGERS]
    if len(reduced_loggers) != 1:
        raise ActivationRefusal("config-shape", "reduced config must retain exactly one CSV logger")
    document["callbacks"] = reduced
    document["loggers"] = reduced_loggers
    # These sections are the scientific contract and must not be translated by
    # the smoke reduction.
    if any(document.get(key) != original_science[key] for key in original_science):
        raise ActivationRefusal("config-translation", "reduced config changed a scientific section")
    effective_text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    diff_text = "".join(
        difflib.unified_diff(
            source_text.splitlines(keepends=True),
            effective_text.splitlines(keepends=True),
            fromfile=str(source),
            tofile=str(output),
        )
    )
    source_hash = _sha256_bytes(source_bytes)
    _atomic_text(output, effective_text)
    diff_path = output.with_suffix(output.suffix + ".diff")
    hash_path = output.with_suffix(output.suffix + ".sha256")
    _atomic_text(diff_path, diff_text)
    effective_hash = file_sha256(output)
    _atomic_text(hash_path, effective_hash + "  " + str(output) + "\n")
    return ReducedConfig(
        str(source),
        str(output),
        str(diff_path),
        str(hash_path),
        source_hash,
        effective_hash,
        cadence,
    )


def _override_values(overrides: Sequence[str]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Parse an explicit override list without silently dropping entries."""

    values: dict[str, str] = {}
    ordered: list[str] = []
    for item in overrides:
        if type(item) is not str or "=" not in item:
            raise ActivationRefusal("train-override", "every override must be a non-empty key=value string")
        key, value = item.split("=", 1)
        if not key or not value or key in values:
            raise ActivationRefusal("train-override", f"override {item!r} is empty or repeated")
        values[key] = value
        ordered.append(item)
    return values, tuple(ordered)


def _declared_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ActivationRefusal("train-fields", f"{name} must be an integer >= {minimum}")
    return value


def _config_value(config: Mapping[str, Any], *keys: str, name: str) -> int:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ActivationRefusal("config-shape", f"missing {name}")
        value = value[key]
    if type(value) is not int or value <= 0:
        raise ActivationRefusal("config-shape", f"{name} must be a positive integer")
    return value


def build_he_train_plan(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    results_root: str | Path,
    source_config: str | Path,
    checkpoint_every_n: int,
    row_ids: Sequence[str] | None = None,
) -> tuple[StagePlanV2, dict[str, tuple[str, ...]]]:
    """Translate selected manifest TRAIN rows into one bounded stage plan.

    Parameters
    ----------
    manifest : mapping
        Producer manifest containing opaque row and plan identities.
    repo_root, results_root, source_config : str or pathlib.Path
        Validated absolute checkout, result root, and source YAML path.
    checkpoint_every_n : int
        Checkpoint cadence bound into the effective config.
    row_ids : sequence of str, optional
        Producer row IDs to select; omitted selects every TRAIN row.

    Returns
    -------
    tuple of StagePlanV2 and dict
        Validated plan and exact per-task executor overrides.

    Raises
    ------
    ActivationRefusal
        If a row is not TPEN TRAIN, binds another config, conflicts with an
        explicit override, or declares unsupported fields.
    """

    root = _absolute(repo_root, "repo_root")
    results = _absolute(results_root, "results_root")
    source = _absolute(source_config, "source_config")
    if not source.is_relative_to(root):
        raise ActivationRefusal("config-path", "source_config must be inside the validated repo_root")
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ActivationRefusal("config-read", f"source config could not be loaded: {error}") from error
    if not isinstance(config, Mapping):
        raise ActivationRefusal("config-shape", "source config must be a mapping")
    config_walkers = _config_value(config, "sampler", "n_walkers", name="sampler.n_walkers")
    default_steps = _config_value(config, "trainer", "max_steps", name="trainer.max_steps")
    config_runtime = config.get("runtime")
    config_seed = 0 if not isinstance(config_runtime, Mapping) else config_runtime.get("seed", 0)
    if type(config_seed) is not int or config_seed < 0:
        raise ActivationRefusal("config-shape", "runtime.seed must be a non-negative integer")
    if type(checkpoint_every_n) is not int or checkpoint_every_n <= 0:
        raise ActivationRefusal("train-override", "checkpoint_every_n must be positive")
    rows = [row for row in manifest.get("rows", ()) if isinstance(row, Mapping) and row.get("kind") == "train"]
    selected = None if row_ids is None else {str(item) for item in row_ids}
    if selected is not None:
        rows = [row for row in rows if str(row.get("row_id")) in selected]
        found = {str(row.get("row_id")) for row in rows}
        missing = sorted(selected - found)
        if missing:
            raise ActivationRefusal("row-selection", f"selected TRAIN rows are absent: {missing}")
    if not rows:
        raise ActivationRefusal("empty-train-selection", "attached He route requires at least one TRAIN row")
    tasks: list[LogicalTaskSpec] = []
    translated: dict[str, tuple[str, ...]] = {}
    for row in rows:
        row_id = str(row.get("row_id"))
        if str(row.get("stage")) != HE_TRAIN_STAGE:
            raise ActivationRefusal("unsupported-stage", f"row {row_id!r} is not a He TRAIN stage row")
        runtime = row.get("runtime", manifest.get("runtime", "tpen-test"))
        if type(runtime) is not str or not runtime.startswith("tpen-"):
            raise ActivationRefusal("unsupported-runtime", f"row {row_id!r} does not declare a TPEN runtime")
        configured = Path(str(row.get("config", source)))
        if not configured.is_absolute():
            configured = root / configured
        if configured.resolve() != source.resolve():
            raise ActivationRefusal("config-binding", f"row {row_id!r} does not bind the selected source config")
        raw_overrides = row.get("overrides", ())
        if isinstance(raw_overrides, (str, bytes, bytearray)) or not isinstance(raw_overrides, Sequence):
            raise ActivationRefusal("train-override", f"row {row_id!r} overrides must be a sequence")
        overrides = tuple(raw_overrides)
        explicit, ordered = _override_values(overrides)
        row_seed = row.get("seed", config_seed)
        row_steps = row.get("max_steps", default_steps)
        row_walkers = row.get("n_walkers", config_walkers)
        seed = _declared_int(row_seed, "runtime.seed", minimum=0)
        steps = _declared_int(row_steps, "trainer.max_steps")
        n_walkers = _declared_int(row_walkers, "sampler.n_walkers")
        owned = {
            "runtime.seed": seed,
            "trainer.max_steps": steps,
            "sampler.n_walkers": n_walkers,
        }
        for key, declared in owned.items():
            if key in explicit:
                try:
                    override_value = int(explicit[key])
                except ValueError as error:
                    raise ActivationRefusal("train-override", f"{key} must be an integer") from error
                if override_value != declared:
                    raise ActivationRefusal(
                        "train-override-conflict",
                        f"row {row_id!r} declares {key}={declared} but overrides it with {override_value}",
                    )
            else:
                ordered += (f"{key}={declared}",)
        cadence = row.get("checkpoint_every_n", row.get("cadence", checkpoint_every_n))
        checkpoint_steps = row.get("checkpoint_steps")
        if checkpoint_steps is not None:
            if isinstance(checkpoint_steps, (str, bytes, bytearray)) or not isinstance(checkpoint_steps, Sequence) or len(checkpoint_steps) != 1:
                raise ActivationRefusal("checkpoint-cadence", f"row {row_id!r} must declare one checkpoint cadence")
            cadence = checkpoint_steps[0]
        if _declared_int(cadence, "checkpoint cadence") != checkpoint_every_n:
            raise ActivationRefusal("checkpoint-cadence", f"row {row_id!r} cadence conflicts with effective config")
        exact = tuple(ordered)
        translated[row_id] = exact
        logical_id = row.get("logical_task_id", row.get("task_id", row_id))
        if type(logical_id) is not str or not logical_id:
            raise ActivationRefusal("task-identity", f"row {row_id!r} has no opaque logical task identity")
        run_id = row.get("run_id", row_id)
        if type(run_id) is not str or not run_id:
            raise ActivationRefusal("task-identity", f"row {row_id!r} has no opaque run identity")
        tasks.append(
            LogicalTaskSpec(
                logical_task_id=logical_id,
                stage=HE_TRAIN_STAGE,
                run_id=run_id,
                command=("tpen", "train"),
                result_dir=str(results / "attached" / row_id),
                params={
                    "kind": "train",
                    "config": str(source),
                    "seed": seed,
                    "n_walkers": n_walkers,
                    "max_steps": steps,
                    "executor_overrides": exact,
                    "checkpoint_every_n": checkpoint_every_n,
                },
                metadata={"route": ROUTE_NAME, "repo_root": str(root), "runtime": runtime},
            ).validate()
        )
    plan_id = manifest.get("plan_id", "attached")
    if type(plan_id) is not str or not plan_id:
        raise ActivationRefusal("plan-identity", "producer plan_id must be a non-empty opaque string")
    plan = StagePlanV2(
        study="tpen_he_v1",
        stage=HE_TRAIN_STAGE,
        plan_id=plan_id,
        results_root=str(results),
        tasks=tuple(tasks),
    ).validate()
    return plan, translated


def prepare_he_train(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    source_config: str | Path,
    enable_attached: bool = False,
    enable_reduced_callbacks: bool = False,
    checkpoint_every_n: int = 25,
    row_ids: Sequence[str] | None = None,
) -> PreparedHeTrain:
    """Prepare an explicitly selected He TRAIN attached generation.

    Parameters
    ----------
    manifest : mapping
        Producer manifest with selected TRAIN rows and plan identity.
    output_dir, repo_root, results_root, source_config : str or pathlib.Path
        Absolute handoff, validated checkout, result root, and source config.
    enable_attached, enable_reduced_callbacks : bool, default=False
        Explicit route and smoke-only callback opt-ins.
    checkpoint_every_n : int, default=25
        Effective Checkpoint cadence.
    row_ids : sequence of str, optional
        Selected producer row IDs.

    Returns
    -------
    PreparedHeTrain
        Prepared plan and fresh handoff identity.

    Raises
    ------
    ActivationRefusal
        If selection, translation, reduction, or handoff generation fails.
        Current sidecars are retired and a diagnostic is preserved.
    """

    if not enable_attached:
        raise ActivationRefusal("attached-opt-in", "He TRAIN attached route requires explicit opt-in")
    handoff_root = _absolute(output_dir, "output_dir")
    try:
        source = _absolute(source_config, "source_config")
        effective = source
        if enable_reduced_callbacks:
            reduced = generate_reduced_callback_config(
                source,
                handoff_root / "effective-he-train.yaml",
                enable_reduced_callbacks=True,
                checkpoint_every_n=checkpoint_every_n,
            )
            effective = Path(reduced.effective_path)
            source_hash = reduced.source_sha256
            effective_hash = reduced.effective_sha256
        else:
            source_hash = file_sha256(source)
            effective_hash = source_hash
        plan, translated = build_he_train_plan(
            manifest,
            repo_root=repo_root,
            results_root=results_root,
            source_config=source,
            checkpoint_every_n=checkpoint_every_n,
            row_ids=row_ids,
        )
    except Exception as error:
        retire_attached_generation(handoff_root, error)
        raise
    handoff = write_attached_handoff(
        handoff_root,
        plan,
        effective_config=effective,
        source_config=source,
        profile=PROFILE_NAME,
        overrides_by_task=translated,
        metadata={
            "route": ROUTE_NAME,
            "stage": "train",
            "source_config_sha256": source_hash,
            "effective_config_sha256": effective_hash,
            "checkpoint_schedule_every_n": checkpoint_every_n,
            "reduced_callbacks": bool(enable_reduced_callbacks),
        },
    )
    return PreparedHeTrain(plan, handoff, str(effective), str(source), checkpoint_every_n)


def activate_he_train(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    source_config: str | Path,
    science_python: str | Path,
    executor_python: str | Path,
    allocation: str | Path,
    worker_config: str | Path,
    store: str | Path,
    receipt_path: str | Path,
    enable_attached: bool = False,
    enable_reduced_callbacks: bool = False,
    checkpoint_every_n: int = 25,
    row_ids: Sequence[str] | None = None,
    dispatcher: str = "parsl",
    timeout: float | None = None,
    cwd: str | Path | None = None,
    inspect_run: Any = None,
    allocation_context: Any = None,
    observed_environment: Mapping[str, str] | None = None,
    observed_hostname: str | None = None,
) -> HeTrainActivation:
    """Run the complete explicit He TRAIN attached activation route.

    Parameters
    ----------
    manifest : mapping
        Producer manifest for the selected TRAIN generation.
    output_dir, repo_root, results_root, source_config : str or pathlib.Path
        Absolute preparation and source paths.
    science_python, executor_python : str or pathlib.Path
        Absolute prepared science and executor Python executables.
    allocation, worker_config, store, receipt_path : str or pathlib.Path
        Allocation, worker, fresh store, and activation receipt paths.
    enable_attached, enable_reduced_callbacks : bool, default=False
        Explicit route and smoke-only callback opt-ins.
    checkpoint_every_n : int, default=25
        Effective Checkpoint cadence.
    row_ids : sequence of str, optional
        Selected producer row IDs.
    dispatcher : str, default="parsl"
        Executor dispatcher backend.
    timeout : float, optional
        Child/evidence timeout.
    cwd : str or pathlib.Path, optional
        Working directory for profile and executor subprocesses.
    inspect_run : callable, optional
        Root public-reader inspector or a test double.
    allocation_context : object, optional
        Validated full attach context to bind into a derived allocation file.
    observed_environment : mapping[str, str], optional
        Environment observed for the allocation-context snapshot.
    observed_hostname : str, optional
        Hostname observed for the allocation-context snapshot.

    Returns
    -------
    HeTrainActivation
        Profile, dispatch, prepared handoff, and receipt result.

    Raises
    ------
    ActivationRefusal
        If preparation, profile, dispatch, allocation binding, or completed
        reader evidence is refused. No retry or fallback is attempted.
    """

    prepared = prepare_he_train(
        manifest,
        output_dir=output_dir,
        repo_root=repo_root,
        results_root=results_root,
        source_config=source_config,
        enable_attached=enable_attached,
        enable_reduced_callbacks=enable_reduced_callbacks,
        checkpoint_every_n=checkpoint_every_n,
        row_ids=row_ids,
    )
    expected = prepared.handoff.task_ids
    allocation_for_dispatch = _absolute(allocation, "allocation")
    if allocation_context is not None:
        try:
            if observed_environment is None or observed_hostname is None:
                raise ActivationRefusal(
                    "allocation-observation",
                    "allocation_context requires observed_environment and observed_hostname",
                )
            allocation_for_dispatch = bind_allocation_context(
                allocation_for_dispatch,
                Path(prepared.handoff.directory) / "allocation-with-context.json",
                allocation_context,
                observed_environment=observed_environment,
                observed_hostname=observed_hostname,
            )
        except ActivationRefusal as error:
            write_activation_failure_receipt(
                receipt_path,
                code=error.code,
                reason=error.reason,
                expected_task_ids=expected,
                manifest_path=Path(store) / "run-manifest.json",
            )
            raise
    try:
        profile = check_profile(science_python, prepared.effective_config, cwd=cwd)
    except ActivationRefusal as error:
        write_activation_failure_receipt(
            receipt_path,
            code=error.code,
            reason=error.reason,
            expected_task_ids=expected,
            manifest_path=Path(store) / "run-manifest.json",
        )
        raise
    dispatch = run_executor_cli(
        executor_python,
        prepared.handoff,
        allocation=allocation_for_dispatch,
        worker_config=worker_config,
        store=store,
        dispatcher=dispatcher,
        timeout=timeout,
        cwd=cwd,
    )
    manifest_path = Path(store) / "run-manifest.json"
    if dispatch.manifest is None or not manifest_path.is_file():
        write_activation_failure_receipt(
            receipt_path,
            code=f"executor-exit-{dispatch.returncode}",
            reason=dispatch.stderr.strip() or "executor did not publish a manifest",
            expected_task_ids=expected,
            profile=profile.profile,
            manifest_path=manifest_path,
        )
        raise ActivationRefusal("executor-failed", f"executor CLI returned {dispatch.returncode}")
    try:
        write_activation_receipt(
            manifest_path,
            receipt_path,
            expected_task_ids=expected,
            inspect_run=inspect_run,
            profile=profile.profile,
        )
    except Exception as error:
        raise ActivationRefusal("reader-failed", str(error)) from error
    return HeTrainActivation(prepared, profile, dispatch, str(receipt_path))


__all__ = [
    "ALLOWED_REDUCED_CALLBACKS",
    "ALLOWED_REDUCED_LOGGERS",
    "HE_TRAIN_STAGE",
    "PreparedHeTrain",
    "HeTrainActivation",
    "REDUCED_CONFIG_SCHEMA",
    "ReducedConfig",
    "ROUTE_NAME",
    "build_he_train_plan",
    "generate_reduced_callback_config",
    "prepare_he_train",
    "activate_he_train",
]
