"""Study-side H-v2 attached-executor activation route (train and eval).

This module is deliberately independent of :mod:`tpen_executor`.  It builds
one :class:`~experiments.toolkit.dispatch.LogicalTaskSpec` per invocation from
``protocol.py``'s frozen configurations (H-v2 has no producer/plan module
yet), checks :func:`protocol.assert_hydrogen_contract` on the exact resolved
config before dispatch, invokes the installed executor CLI as a separate
subprocess through :mod:`experiments.toolkit.attached_activation`, and reads
results only through :func:`experiments.toolkit.run_results.read`.  It never
imports ``tpen_executor`` and reaches core only through configuration or the
``tpen`` command, following ``experiments/README.md``.

``tpen-executor/run-manifest-v2`` is a closed record that cannot carry H-v2
science identity, so this module additionally writes a study-side accepted
receipt (:func:`write_h_v2_accepted_receipt`) keyed per accepted segment,
holding the resolved-config hash, the trajectory hash when one exists, the
source commit, and an open, versioned artifact map that later logging layers
extend without redefining this schema.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from omegaconf import OmegaConf

from experiments.toolkit.attached_activation import (
    PROFILE_NAME,
    ActivationRefusal,
    AttachedHandoff,
    ExecutorDispatchResult,
    ProfileVerdict,
    check_profile,
    file_sha256,
    run_executor_cli,
    write_activation_failure_receipt,
    write_activation_receipt,
    write_attached_handoff,
)
from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2
from experiments.toolkit.run_results import ManifestRefusal, _default_inspect_run, read as read_manifest

HERE = Path(__file__).parent


def _h_v2_module(name: str) -> Any:
    """Load an h-v2 sibling under the same ``h_v2_<name>`` key its tests use.

    ``experiments/`` has several same-named ``protocol.py``/``collect.py``
    files across studies, so a bare ``import`` would silently resolve to
    whichever study's copy loaded first.  ``test_protocol.py`` and
    ``test_collect.py`` already load this directory's siblings under an
    explicit ``h_v2_<name>`` :mod:`sys.modules` key; reusing that key here
    keeps every h-v2 file sharing one cached module instead of introducing a
    third, inconsistent loading path.
    """

    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _h_v2_module("protocol")


ROUTE_NAME = "h-v2-attached"
TRAIN_STAGE = "train"
EVAL_STAGE = "eval"
H_V2_ACCEPTED_RECEIPT_SCHEMA = "tpen-experiments/h-v2-accepted-receipt-v1"
#: F1: EVAL cannot yet produce accepted checkpoint-based evidence.  EVAL's
#: model-only restore writes no training checkpoint, so
#: ``tpen/inspection.py``'s ``inspect_run`` returns ``missing-checkpoint``
#: even for a run that completed correctly, and the executor's own contract
#: (``executor/src/tpen_executor/dispatch/contract.py``) publishes no
#: accepted segment without checkpoint evidence.  This is an inter-lane
#: core/executor contract decision (see TO item 9d3aa0a3, note
#: ``blocker-recovery-profile-callback-whitelist-20261109``), not something
#: this study-side module can resolve by inventing a training checkpoint.
EVAL_ROUTE_UNSUPPORTED_CODE = "h-v2-eval-route-unsupported"
EVAL_ROUTE_UNSUPPORTED_REASON = (
    "H-v2 EVAL cannot yet produce accepted checkpoint-based evidence: a model-only "
    "restore writes no training checkpoint, so tpen/inspection.py's inspect_run "
    "returns 'missing-checkpoint' for a run that completed correctly, and the "
    "executor's dispatch contract publishes no accepted segment without one. "
    "Blocked on an inter-lane core/executor contract decision for completed-EVAL "
    "evidence; see TO item 9d3aa0a3-0f59-43e6-b52b-dcf39a39785d."
)
#: Artifacts present in every segment's core run directory (both callback
#: configs declare ``ResolvedConfigSnapshot``/``Metadata`` unconditionally).
_BASE_ARTIFACTS = {"resolved_config": "resolved_config.yaml", "metadata": "metadata.json"}
#: Trajectory artifacts, required for every EVAL segment and never declared
#: for TRAIN (``SampledRecordWriter``'s envelope, written under the real
#: frozen ``configs/eval.yaml:129`` evaluator task output directory,
#: ``${run.dir}/trajectory``, i.e. ``trajectory/`` relative to the fenced
#: core run directory).
_TRAJECTORY_ARTIFACTS = {
    "trajectory_csv": "trajectory/sampled_eval_table.csv",
    "trajectory_metadata": "trajectory/sampled_eval_table.metadata.json",
}


class HV2ReceiptRefusal(ValueError):
    """A named refusal while building the study-side H-v2 accepted receipt.

    Parameters
    ----------
    code : str
        Stable machine-readable refusal category.
    reason : str
        Human-readable explanation of the rejected segment or evidence.
    """

    def __init__(self, code: str, reason: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("refusal code must be a non-empty string")
        if type(reason) is not str or not reason:
            raise TypeError("refusal reason must be a non-empty string")
        self.code = code
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PreparedHV2Run:
    """One prepared H-v2 attached generation, before profile/CLI execution."""

    plan: StagePlanV2
    handoff: AttachedHandoff
    task_id: str
    effective_config: str
    config: Mapping[str, Any]
    overrides: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HV2Activation:
    """The operational CLI result plus both receipt paths."""

    prepared: PreparedHV2Run
    profile: ProfileVerdict
    dispatch: ExecutorDispatchResult
    activation_receipt_path: str
    h_v2_receipt_path: str


def _absolute(value: str | Path, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise HV2ReceiptRefusal("config-path", f"{name} must be absolute")
    return path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_h_v2_train_plan(
    *,
    output_dir: str | Path,
    results_root: str | Path,
    lineage: str,
    seed: int | None = None,
    n_walkers: int | None = None,
    max_steps: int | None = None,
) -> tuple[StagePlanV2, dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build the one-task H-v2 TRAIN plan from the frozen training config.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Absolute directory the materialized config YAML is written into.
    results_root : str or pathlib.Path
        Absolute root the task's run output is placed under.
    lineage : str
        Opaque lineage identity (for example ``"L1"``); becomes part of the
        task identity and the run's ``run.run_id``.
    seed, n_walkers, max_steps : int, optional
        Overrides of the frozen training config's literals.  Omitted values
        default to the config's own values, so a bare call reproduces the
        frozen pipeline-check config exactly.

    Returns
    -------
    tuple of StagePlanV2, dict, dict
        The one-task plan, ``{task_id: executor_overrides}``, and the exact
        resolved config dict that was checked against the hydrogen contract
        and written to disk.

    Raises
    ------
    HV2ReceiptRefusal
        If ``output_dir`` or ``results_root`` is not absolute.
    ValueError
        If the resolved config fails ``protocol.assert_hydrogen_contract``.
    """

    out = _absolute(output_dir, "output_dir")
    results = _absolute(results_root, "results_root")
    config = protocol.training_config()
    seed = config["runtime"]["seed"] if seed is None else seed
    n_walkers = config["sampler"]["n_walkers"] if n_walkers is None else n_walkers
    max_steps = config["trainer"]["max_steps"] if max_steps is None else max_steps
    config["runtime"]["seed"] = seed
    config["sampler"]["n_walkers"] = n_walkers
    config["trainer"]["max_steps"] = max_steps
    protocol.assert_hydrogen_contract(config, evaluation=False)
    task_id = f"h-v2-train-{lineage}"
    result_dir = results / "attached" / task_id
    config["run"]["root"] = str(result_dir.parent)
    config["run"]["run_id"] = task_id
    config["run"]["layout"] = "flat"
    config_path = out / f"{task_id}.yaml"
    _atomic_text(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    overrides = (
        "run.layout=flat",
        f"run.root={result_dir.parent}",
        f"run.run_id={task_id}",
        f"runtime.seed={seed}",
        f"trainer.max_steps={max_steps}",
        f"sampler.n_walkers={n_walkers}",
    )
    task = LogicalTaskSpec(
        logical_task_id=task_id,
        stage=TRAIN_STAGE,
        run_id=task_id,
        command=("tpen", "train"),
        result_dir=str(result_dir),
        params={
            "kind": "train",
            "config": str(config_path),
            "seed": seed,
            "n_walkers": n_walkers,
            "max_steps": max_steps,
            "executor_overrides": overrides,
        },
        metadata={"route": ROUTE_NAME, "lineage": lineage},
    ).validate()
    plan = StagePlanV2(
        study="tpen_h_v2",
        stage=TRAIN_STAGE,
        plan_id=task_id,
        results_root=str(results),
        tasks=(task,),
    ).validate()
    return plan, {task_id: overrides}, config


def build_h_v2_eval_plan(
    *,
    output_dir: str | Path,
    results_root: str | Path,
    lineage: str,
    checkpoint: str | Path,
    stream: int = 0,
    seed: int | None = None,
) -> tuple[StagePlanV2, dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build the one-task H-v2 EVAL plan restoring a trained checkpoint.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Absolute directory the materialized config YAML is written into.
    results_root : str or pathlib.Path
        Absolute root the task's run output is placed under.
    lineage : str
        Opaque lineage identity; becomes part of the task identity.
    checkpoint : str or pathlib.Path
        Absolute checkpoint directory produced by the lineage's TRAIN task
        (read from that task's accepted receipt by the caller, never
        invented here).
    stream : {0, 1}, default=0
        Fresh-evaluation stream selecting ``protocol.EVALUATION_SEEDS``.
    seed : int, optional
        Explicit seed override; defaults to the stream's frozen seed.

    Returns
    -------
    tuple of StagePlanV2, dict, dict
        The one-task plan, ``{task_id: executor_overrides}``, and the exact
        resolved config dict that was checked against the hydrogen contract
        and written to disk.

    Raises
    ------
    HV2ReceiptRefusal
        If ``output_dir``, ``results_root``, or ``checkpoint`` is not
        absolute.
    ValueError
        If the resolved config fails ``protocol.assert_hydrogen_contract``.
    """

    out = _absolute(output_dir, "output_dir")
    results = _absolute(results_root, "results_root")
    checkpoint_dir = _absolute(checkpoint, "checkpoint")
    task_id = f"h-v2-eval-{lineage}-stream{stream}"
    result_dir = results / "attached" / task_id
    config = protocol.evaluation_config(checkpoint_dir, stream=stream, root=result_dir.parent)
    resolved_seed = config["runtime"]["seed"] if seed is None else seed
    config["runtime"]["seed"] = resolved_seed
    config["run"]["run_id"] = task_id
    config["run"]["layout"] = "flat"
    protocol.assert_hydrogen_contract(config, evaluation=True)
    config_path = out / f"{task_id}.yaml"
    _atomic_text(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    overrides = (
        "run.layout=flat",
        f"run.root={result_dir.parent}",
        f"run.run_id={task_id}",
        f"runtime.seed={resolved_seed}",
        f"load.path={checkpoint_dir}",
    )
    task = LogicalTaskSpec(
        logical_task_id=task_id,
        stage=EVAL_STAGE,
        run_id=task_id,
        command=("tpen", "eval"),
        result_dir=str(result_dir),
        params={
            "kind": "eval",
            "config": str(config_path),
            "seed": resolved_seed,
            "checkpoint_dir": str(checkpoint_dir),
            "n_walkers": config["sampler"]["n_walkers"],
            "n_draws": config["evaluator"]["tasks"][0]["generator"]["n_draws"],
            "burn_in": config["sampler"]["burn_in"],
            "discard_draws": config["evaluator"]["tasks"][0]["generator"]["discard_draws"],
            "stride": 0,
            "chunk_size": config["evaluator"]["tasks"][0]["generator"]["chunk_size"],
            "task_names": tuple(task_entry["name"] for task_entry in config["evaluator"]["tasks"]),
            "executor_overrides": overrides,
        },
        metadata={"route": ROUTE_NAME, "lineage": lineage, "stream": stream},
    ).validate()
    plan = StagePlanV2(
        study="tpen_h_v2",
        stage=EVAL_STAGE,
        plan_id=task_id,
        results_root=str(results),
        tasks=(task,),
    ).validate()
    return plan, {task_id: overrides}, config


def _compose_config_with_overrides(effective_config: str | Path, overrides: Sequence[str]) -> dict[str, Any]:
    """Compose the exact config that will launch: the effective file plus overrides.

    Mirrors ``experiments.toolkit.attached_activation.check_profile``'s own
    composition (``OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))``)
    so the contract check and the profile check see the same bound config.
    """

    base = OmegaConf.load(effective_config)
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    composed = OmegaConf.to_container(merged, resolve=False)
    assert isinstance(composed, dict)
    return composed


def prepare_h_v2_run(
    plan: StagePlanV2,
    overrides_by_task: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> PreparedHV2Run:
    """Write the attached handoff for a one-task H-v2 plan.

    Parameters
    ----------
    plan : StagePlanV2
        Plan returned by :func:`build_h_v2_train_plan` or
        :func:`build_h_v2_eval_plan`.
    overrides_by_task : mapping[str, sequence[str]]
        Exact per-task override tuple returned alongside ``plan``.
    config : mapping
        The exact resolved config dict returned alongside ``plan``, used
        later to bind stage-specific artifact declarations to the prepared
        config (see :func:`activate_h_v2_run`).
    output_dir : str or pathlib.Path
        Absolute handoff directory.

    Returns
    -------
    PreparedHV2Run
        Prepared plan and fresh handoff identity.

    Raises
    ------
    HV2ReceiptRefusal
        If the plan does not carry exactly one task.
    ValueError
        If the exact bound config -- the effective config file composed with
        its own overrides, the same composition that will actually launch --
        fails :func:`protocol.assert_hydrogen_contract`.  Carried requirement
        1 is checked here, on the COMPOSED config, not only on the builder's
        unresolved dict: an override supplied after the builder ran (for
        example a tampered ``system.n_particles=2``) is caught here, before
        any dispatch.
    ActivationRefusal
        If handoff translation fails; current sidecars are retired and a
        diagnostic is preserved (see ``write_attached_handoff``).
    """

    if len(plan.tasks) != 1:
        raise HV2ReceiptRefusal("multi-task-plan", "the H-v2 attached route dispatches exactly one task")
    task = plan.tasks[0]
    effective_config = task.params["config"]
    overrides = tuple(str(item) for item in overrides_by_task[task.logical_task_id])
    composed = _compose_config_with_overrides(effective_config, overrides)
    protocol.assert_hydrogen_contract(composed, evaluation=(task.stage == EVAL_STAGE))
    handoff = write_attached_handoff(
        output_dir,
        plan,
        effective_config=effective_config,
        source_config=effective_config,
        overrides_by_task=overrides_by_task,
        metadata={"route": ROUTE_NAME, "stage": task.stage},
    )
    return PreparedHV2Run(plan, handoff, task.logical_task_id, str(effective_config), dict(config), overrides)


def activate_h_v2_run(
    prepared: PreparedHV2Run,
    *,
    science_python: str | Path,
    executor_python: str | Path,
    allocation: str | Path,
    worker_config: str | Path,
    store: str | Path,
    activation_receipt_path: str | Path,
    h_v2_receipt_path: str | Path,
    expected_source_commit: str,
    dispatcher: str = "local",
    timeout: float | None = None,
    cwd: str | Path | None = None,
    inspect_run: Any = None,
    artifact_declarations: Mapping[str, str] | None = None,
) -> HV2Activation:
    """Run the complete H-v2 attached activation route for one prepared task.

    Parameters
    ----------
    prepared : PreparedHV2Run
        Value returned by :func:`prepare_h_v2_run`.
    science_python, executor_python : str or pathlib.Path
        Absolute prepared science and executor Python executables.
    allocation, worker_config, store : str or pathlib.Path
        Allocation, worker, and fresh store paths.
    activation_receipt_path, h_v2_receipt_path : str or pathlib.Path
        Destinations for the generic P10 receipt and the H-v2 accepted
        receipt.
    expected_source_commit : str
        The exact commit this job checked out (for example ``git rev-parse
        HEAD`` inside the job); compared against every segment's recorded
        source commit.
    dispatcher : {"local", "parsl"}, default="local"
        Executor dispatcher backend.
    timeout : float, optional
        Child/evidence timeout.
    cwd : str or pathlib.Path, optional
        Working directory for profile and executor subprocesses.
    inspect_run : callable, optional
        Root public-reader inspector, or a test double.
    artifact_declarations : mapping[str, str], optional
        Additional ``{name: relative_path}`` entries, beyond the base and
        trajectory artifacts this function already declares, that every
        accepted segment must carry.  Later logging layers extend this
        without redefining the receipt schema.

    Returns
    -------
    HV2Activation
        Profile, dispatch, and both receipt paths.

    Raises
    ------
    ActivationRefusal
        If the profile check or executor dispatch is refused.
    HV2ReceiptRefusal, ManifestRefusal
        If the H-v2 accepted receipt refuses the run (see
        :func:`build_h_v2_accepted_receipt`).  Both the generic and the
        H-v2 failure receipts are written before the exception propagates.
    """

    expected = prepared.handoff.task_ids
    manifest_path = Path(store) / "run-manifest.json"
    stage = prepared.plan.tasks[0].stage

    def _fail(code: str, reason: str, *, profile: str | None = None) -> None:
        for path in (activation_receipt_path, h_v2_receipt_path):
            write_activation_failure_receipt(
                path,
                code=code,
                reason=reason,
                expected_task_ids=expected,
                profile=profile if profile is not None else PROFILE_NAME,
                manifest_path=manifest_path,
            )

    if stage == EVAL_STAGE:
        _fail(EVAL_ROUTE_UNSUPPORTED_CODE, EVAL_ROUTE_UNSUPPORTED_REASON)
        raise ActivationRefusal(EVAL_ROUTE_UNSUPPORTED_CODE, EVAL_ROUTE_UNSUPPORTED_REASON)

    try:
        profile = check_profile(science_python, prepared.effective_config, overrides=prepared.overrides, cwd=cwd)
    except ActivationRefusal as error:
        _fail(error.code, error.reason)
        raise
    dispatch = run_executor_cli(
        executor_python,
        prepared.handoff,
        allocation=allocation,
        worker_config=worker_config,
        store=store,
        dispatcher=dispatcher,
        timeout=timeout,
        cwd=cwd,
    )
    if dispatch.manifest is None or not manifest_path.is_file():
        code = f"executor-exit-{dispatch.returncode}"
        reason = dispatch.stderr.strip() or "executor did not publish a manifest"
        _fail(code, reason, profile=profile.profile)
        raise ActivationRefusal("executor-failed", f"executor CLI returned {dispatch.returncode}")
    try:
        write_activation_receipt(
            manifest_path,
            activation_receipt_path,
            expected_task_ids=expected,
            inspect_run=inspect_run,
            profile=profile.profile,
        )
    except Exception as error:
        write_activation_failure_receipt(
            h_v2_receipt_path,
            code="reader-failed",
            reason=str(error),
            expected_task_ids=expected,
            profile=profile.profile,
            manifest_path=manifest_path,
        )
        raise ActivationRefusal("reader-failed", str(error)) from error
    try:
        write_h_v2_accepted_receipt(
            manifest_path,
            h_v2_receipt_path,
            expected_task_ids=expected,
            expected_source_commit=expected_source_commit,
            stage=stage,
            inspect_run=inspect_run,
            artifact_declarations=artifact_declarations,
        )
    except (HV2ReceiptRefusal, ManifestRefusal) as error:
        write_activation_failure_receipt(
            h_v2_receipt_path,
            code=error.code,
            reason=error.reason,
            expected_task_ids=expected,
            profile=profile.profile,
            manifest_path=manifest_path,
        )
        raise
    except Exception as error:
        write_activation_failure_receipt(
            h_v2_receipt_path,
            code="h-v2-receipt-build-failed",
            reason=str(error),
            expected_task_ids=expected,
            profile=profile.profile,
            manifest_path=manifest_path,
        )
        raise
    return HV2Activation(prepared, profile, dispatch, str(activation_receipt_path), str(h_v2_receipt_path))


def build_h_v2_accepted_receipt(
    manifest_path: str | Path,
    *,
    expected_task_ids: Sequence[str],
    expected_source_commit: str,
    stage: str = TRAIN_STAGE,
    inspect_run: Any = None,
    artifact_declarations: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the study-side H-v2 accepted-receipt payload from one manifest.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Published executor manifest; read only through
        :func:`experiments.toolkit.run_results.read`, never hand-parsed.
    expected_task_ids : sequence of str
        Exact ordered task identities expected from this generation.
    expected_source_commit : str
        The exact commit this job checked out.
    stage : {"train", "eval"}, default="train"
        Segment stage, bound to the prepared config by the caller (see
        :func:`activate_h_v2_run`).  EVAL segments additionally and
        unconditionally declare the trajectory pair, written under the real
        frozen ``configs/eval.yaml:129`` evaluator task output directory
        (``trajectory/`` relative to the fenced core run directory).  TRAIN
        segments never produce a trajectory and never declare one, so
        ``trajectory_csv_sha256`` stays ``None`` for them.
    inspect_run : callable, optional
        Public core inspector, or a test double receiving each fenced core
        run directory.
    artifact_declarations : mapping[str, str], optional
        Additional ``{name: relative_path}`` entries every accepted segment
        must carry, beyond the base (``resolved_config``, ``metadata``) and,
        for EVAL, the trajectory artifacts this function already declares.
        Later logging layers extend this without redefining the receipt
        schema.

    Returns
    -------
    dict
        The ``tpen-experiments/h-v2-accepted-receipt-v1`` payload, keyed per
        accepted segment by ``(task_id, attempt_id, checkpoint_dir)``.

    Raises
    ------
    ManifestRefusal
        If the manifest, chain, or public core evidence is inconsistent
        (propagated unchanged from ``read()``), the task-set does not match
        ``expected_task_ids``, or the recorded trajectory has an
        uncommitted tail.
    HV2ReceiptRefusal
        If a segment's source commit disagrees with ``expected_source_commit``,
        a chain's segments disagree on their resolved-config hash (a wrong
        resolved-config identity drifting mid-chain), or a declared artifact
        -- including both trajectory entries for an EVAL segment -- is
        missing.

    Notes
    -----
    This function does not carry a trusted, independently-computed expected
    ``resolved_config_sha256`` per segment (amendment item 3's
    ``collect.validate_run_identity`` reconciliation).  ``resolved_config.yaml``
    is written by ``tpen.callback.ResolvedConfigSnapshot`` via
    ``OmegaConf.save(context.cfg, path, resolve=True)``, where ``context.cfg``
    additionally carries ``run.root``/``run.run_id``/``run.layout`` values the
    ATTACHED EXECUTOR forces onto every dispatched command
    (``executor/src/tpen_executor/dispatch/contract.py``,
    ``dispatch/parsl.py``), overriding this module's own ``run.root``/
    ``run.run_id`` overrides.  That forcing is executor-private behavior, not
    part of any publicly documented toolkit contract (the only public
    contract is ``experiments/toolkit/run_results.py``'s ``"/core"`` fencing
    suffix); reproducing it here to compute an independent expected hash
    would couple this study module to undocumented executor internals rather
    than to a supported boundary, and risk silent, wrong refusals the moment
    that internal forcing changes.  Per orchestrator decision F4, this is
    reported rather than worked around: the cross-segment resolved-config
    equality check below is retained as defense-in-depth for a multi-segment
    chain, but a single-segment run (the normal case for this route) has no
    independent identity check.  See TO item
    9d3aa0a3-0f59-43e6-b52b-dcf39a39785d.
    """

    if stage not in (TRAIN_STAGE, EVAL_STAGE):
        raise ValueError(f"stage must be {TRAIN_STAGE!r} or {EVAL_STAGE!r}, got {stage!r}")
    manifest_path = Path(manifest_path)
    expected = tuple(str(item) for item in expected_task_ids)
    declared = dict(_BASE_ARTIFACTS)
    if stage == EVAL_STAGE:
        declared.update(_TRAJECTORY_ARTIFACTS)
    if artifact_declarations:
        declared.update(artifact_declarations)

    recorded_run_dirs: list[Path] = []

    def completed_inspector(path: str | Path) -> Any:
        run_dir = Path(path)
        evidence = inspect_run(run_dir) if inspect_run is not None else _default_inspect_run(run_dir)
        verdict = evidence.get("verdict") if isinstance(evidence, Mapping) else getattr(evidence, "verdict", None)
        run_status = (
            evidence.get("run_status") if isinstance(evidence, Mapping) else getattr(evidence, "run_status", None)
        )
        if verdict != "completed" or run_status != "completed":
            raise ManifestRefusal(
                "core-run-incomplete", "H-v2 acceptance requires public core verdict and run_status completed"
            )
        recorded_run_dirs.append(run_dir)
        return evidence

    result = read_manifest(manifest_path, inspect_run=completed_inspector)
    actual = tuple(chain.task_id for chain in result.chains)
    if actual != expected:
        raise ManifestRefusal("task-set-mismatch", f"reader chains {actual!r} differ from expected {expected!r}")
    if not result.chains or any(not chain.segments for chain in result.chains):
        raise ManifestRefusal("empty-chain", "reader returned no nonempty committed chain")

    # read() calls the inspector exactly once per accepted segment, in the
    # manifest's sorted (task_id, next_iteration, completed_updates) order,
    # which groups same-task segments contiguously; result.chains preserves
    # that same first-occurrence order. Flattening both the same way lines
    # the recorded run directories back up with their segments.
    flat_segments = [(chain.task_id, segment) for chain in result.chains for segment in chain.segments]

    segments_payload: list[dict[str, Any]] = []
    for (task_id, segment), core_dir in zip(flat_segments, recorded_run_dirs, strict=True):
        # Every declared artifact -- base, trajectory (EVAL only), and any
        # caller-supplied additions -- is required unconditionally: a
        # declared file that is missing refuses the whole segment rather
        # than being silently skipped.
        artifacts: dict[str, dict[str, str]] = {}
        for name, relative in declared.items():
            artifact_path = core_dir / relative
            if not artifact_path.is_file():
                raise HV2ReceiptRefusal("missing-artifact", f"{task_id!r} is missing declared artifact {name!r}")
            artifacts[name] = {"path": str(artifact_path), "sha256": file_sha256(artifact_path)}

        metadata_path = core_dir / declared["metadata"]
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_commit = run_metadata.get("git_commit")
        if actual_commit != expected_source_commit:
            raise HV2ReceiptRefusal(
                "wrong-source-commit",
                f"{task_id!r} segment {segment.attempt_id!r} ran commit {actual_commit!r}, "
                f"expected {expected_source_commit!r}",
            )
        resolved_config_sha256 = artifacts["resolved_config"]["sha256"]

        trajectory_csv_sha256: str | None = None
        if "trajectory_csv" in declared:
            trajectory_metadata_path = core_dir / declared["trajectory_metadata"]
            trajectory_manifest = json.loads(trajectory_metadata_path.read_text(encoding="utf-8"))
            trajectory_csv_sha256 = trajectory_manifest.get("csv_sha256")
            if trajectory_csv_sha256 != artifacts["trajectory_csv"]["sha256"]:
                raise ManifestRefusal(
                    "uncommitted-tail", f"{task_id!r} segment {segment.attempt_id!r} trajectory csv is stale"
                )

        segments_payload.append(
            {
                "task_id": task_id,
                "attempt_id": segment.attempt_id,
                "checkpoint_dir": segment.checkpoint_dir,
                "resolved_config_sha256": resolved_config_sha256,
                "trajectory_csv_sha256": trajectory_csv_sha256,
                "source_commit": actual_commit,
                "artifacts": artifacts,
            }
        )

    by_task: dict[str, list[str]] = {}
    for payload in segments_payload:
        by_task.setdefault(payload["task_id"], []).append(payload["resolved_config_sha256"])
    for task_id, hashes in by_task.items():
        if len(set(hashes)) != 1:
            raise HV2ReceiptRefusal(
                "wrong-resolved-config-hash", f"{task_id!r} segments disagree on their resolved_config_sha256"
            )

    return {
        "schema": H_V2_ACCEPTED_RECEIPT_SCHEMA,
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": file_sha256(manifest_path),
        "segments": segments_payload,
        "retry": "forbidden",
    }


def write_h_v2_accepted_receipt(
    manifest_path: str | Path,
    receipt_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and atomically publish the H-v2 accepted receipt.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Published executor manifest.
    receipt_path : str or pathlib.Path
        Destination for the durable receipt.
    **kwargs
        Forwarded to :func:`build_h_v2_accepted_receipt`.

    Returns
    -------
    dict
        The payload that was written.

    Raises
    ------
    ManifestRefusal, HV2ReceiptRefusal
        Propagated unchanged from :func:`build_h_v2_accepted_receipt`; no
        receipt is written on refusal (the caller writes a separate failure
        receipt, mirroring ``write_activation_failure_receipt``).
    """

    payload = build_h_v2_accepted_receipt(manifest_path, **kwargs)
    _atomic_text(Path(receipt_path), json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return payload


__all__ = [
    "EVAL_ROUTE_UNSUPPORTED_CODE",
    "EVAL_ROUTE_UNSUPPORTED_REASON",
    "EVAL_STAGE",
    "H_V2_ACCEPTED_RECEIPT_SCHEMA",
    "HV2Activation",
    "HV2ReceiptRefusal",
    "PreparedHV2Run",
    "ROUTE_NAME",
    "TRAIN_STAGE",
    "activate_h_v2_run",
    "build_h_v2_accepted_receipt",
    "build_h_v2_eval_plan",
    "build_h_v2_train_plan",
    "prepare_h_v2_run",
    "write_h_v2_accepted_receipt",
]
