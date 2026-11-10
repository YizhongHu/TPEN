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

import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedInterpolationType

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
#: TO baff945c: required science-config identity fields a payload=science
#: receipt must carry (see :func:`build_h_v2_accepted_receipt`'s refusal and
#: ``run_job.checkpoint_dir_for_eval``'s consumer-side check).
SCIENCE_CONFIG_IDENTITY_KEYS = frozenset(
    {
        "lineage",
        "lineage_index",
        "overlay_sha256",
        "base_train_sha256",
        "contract_inputs_sha256",
        "materialized_config_sha256",
    }
)
#: K1: required science-config identity fields a pilot/part-(g) receipt must
#: carry (``protocol.pilot_training_config``'s own identity-dict shape,
#: which differs from the science payload's -- ``payload``/``seed_namespace``/
#: ``scale_overlay_sha256``/``payload_overlay_sha256`` rather than
#: ``lineage_index``/``overlay_sha256``). A receipt refuses rather than being
#: written with any of these silently dropped between the protocol builder,
#: ``build_h_v2_train_plan``, and this receipt.
PILOT_CONFIG_IDENTITY_KEYS = frozenset(
    {
        "payload",
        "lineage",
        "seed_namespace",
        "scale_overlay_sha256",
        "payload_overlay_sha256",
        "base_train_sha256",
        "contract_inputs_sha256",
        "materialized_config_sha256",
    }
)

#: D3: the unresolved ``${run.dir}/...`` prefix every per-run artifact path in
#: these frozen configs is written under; stripping it from a callback's own
#: unresolved config value gives the artifact's path relative to the fenced
#: core run directory, without resolving ``run.dir`` itself (which is not yet
#: known at prepare time -- the attached executor assigns it, see
#: ``build_h_v2_accepted_receipt``'s docstring).
_RUN_DIR_PREFIX = "${run.dir}/"


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
    artifact_declarations: Mapping[str, str]
    payload: str = "pipeline-check"
    science_config_identity: Mapping[str, Any] | None = None


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


#: Run via ``science_python -c <this> <probe_config_path> <target>``: loads
#: the unresolved probe config, finds *target*'s own callback node, and
#: instantiates ONLY that node against its real constructor -- never the
#: whole config (no model/context exists at probe time). A syntax-only
#: profile admission (:func:`check_profile`) cannot by itself tell "this
#: _target_ is allowlisted" apart from "this exact argument set constructs
#: the real class" (amendment r4 finding: ENDQ/ModuleNorms both passed
#: admission on dev while carrying the wrong/missing constructor arguments).
_INSTANTIATE_CANDIDATE_SCRIPT = """
import sys
from hydra.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
target = sys.argv[2]
node = next((cb for cb in config.get("callbacks", []) if cb.get("_target_") == target), None)
if node is None:
    print(f"MISSING:{target}")
    raise SystemExit(1)
try:
    instantiate(node)
except Exception as error:  # noqa: BLE001 - reported to the caller, never swallowed
    print(f"INSTANTIATION_FAILED:{type(error).__name__}:{error}")
    raise SystemExit(1)
print("OK")
"""


def assert_candidate_instantiates(
    science_python: str | Path,
    probe_config_path: str | Path,
    target: str,
    *,
    cwd: str | Path | None = None,
) -> None:
    """Instantiate *target*'s candidate node against its real constructor.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``candidate-module-misconfigured``, if *target*'s node is
        missing from the probe config or fails to construct. This is
        ALWAYS a hard failure, never a skip: a candidate the recovery
        profile admits (a syntax check) but whose arguments do not match
        its real signature is a configuration bug in this study module,
        not an absent core capability.
    """

    result = subprocess.run(
        [str(science_python), "-c", _INSTANTIATE_CANDIDATE_SCRIPT, str(probe_config_path), target],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        raise HV2ReceiptRefusal(
            "candidate-module-misconfigured",
            f"{target} is admitted by the recovery profile but does not instantiate against its "
            f"real constructor: {result.stdout.strip()!r} {result.stderr.strip()!r}",
        )


def probe_admitted_a1_modules(
    config: dict[str, Any],
    candidates: Sequence[str],
    *,
    science_python: str | Path,
    probe_config_path: str | Path,
    cwd: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Enable every candidate A1 training-side module the recovery profile admits.

    TO 63690220 amendment r3/r4: tries each of *candidates* in order,
    appending it to *config* (:func:`protocol.append_candidate_module`) and
    probing the runtime recovery profile (:func:`check_profile`, which
    subprocesses the exact *science_python* this job will actually train
    with -- never assumed from the branch); a refused candidate is removed
    again (:func:`protocol.remove_candidate_module`) before the next is
    tried, so *config* ends up carrying exactly the admitted subset.
    Skipping is never silent: the caller records both lists on the TRAIN
    receipt.

    Amendment r4: profile admission is syntax-only (an allowlisted
    ``_target_`` plus safe cadence fields) and cannot by itself tell a
    genuinely absent capability apart from a candidate built with the wrong
    arguments for its real constructor. Every admitted candidate is
    therefore ALSO instantiated against its real class
    (:func:`assert_candidate_instantiates`) before being recorded as
    enabled. ONLY an :class:`ActivationRefusal` from :func:`check_profile`
    counts as "not admitted"; any other error -- including an instantiation
    failure -- propagates and is never recorded as a skip.

    Parameters
    ----------
    config : dict
        Mutated in place: ends up carrying exactly the admitted candidates.
    candidates : sequence of str
        Candidate callback targets, tried in order
        (:data:`protocol.A1_TRAINING_MODULE_CANDIDATES`).
    science_python : str or pathlib.Path
        Exact job interpreter the real TRAIN dispatch will use.
    probe_config_path : str or pathlib.Path
        Scratch file this function repeatedly overwrites with *config*'s
        current state; never the task's own materialized config path.

    Returns
    -------
    tuple of list, list
        ``(enabled, skipped_not_admitted)``, in candidate order.
    """

    enabled: list[str] = []
    skipped: list[str] = []
    probe_path = Path(probe_config_path)
    _atomic_text(probe_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    # F3: the base config (no candidate appended yet) must itself be
    # admitted. A failure here is a checker/environment problem, never a
    # per-candidate "not admitted" verdict, so it is never caught -- it
    # propagates unconditionally, including a ``profile-refused`` verdict.
    check_profile(science_python, probe_path, cwd=cwd)
    for target in candidates:
        protocol.append_candidate_module(config, target)
        _atomic_text(probe_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        try:
            check_profile(science_python, probe_path, cwd=cwd)
        except ActivationRefusal as error:
            # F3: only a genuine checker REFUSAL of this candidate is a skip.
            # Any other code (malformed/unavailable/mismatched checker
            # output) is a checker failure, not an admission decision, and
            # must propagate rather than silently drop a valid candidate.
            if error.code != "profile-refused":
                raise
            protocol.remove_candidate_module(config, target)
            skipped.append(target)
            continue
        assert_candidate_instantiates(science_python, probe_path, target, cwd=cwd)
        enabled.append(target)
    return enabled, skipped


def build_h_v2_train_plan(
    *,
    output_dir: str | Path,
    results_root: str | Path,
    lineage: str,
    seed: int | None = None,
    n_walkers: int | None = None,
    max_steps: int | None = None,
    config: Mapping[str, Any] | None = None,
    science_config_identity: Mapping[str, Any] | None = None,
    payload_label: str | None = None,
) -> tuple[StagePlanV2, dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build the one-task H-v2 TRAIN plan from a training config.

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
        Overrides of ``config``'s literals.  Omitted values default to
        ``config``'s own values, so a bare call reproduces whichever base
        config was passed in exactly -- this never reapplies the frozen
        pipeline-check config's own defaults on top of a different ``config``.
    config : mapping, optional
        Base config to build from. Defaults to
        ``protocol.training_config()`` (the frozen pipeline-check config).
        A caller building the science config passes
        ``protocol.science_training_config(...)``'s resolved config here
        instead.
    science_config_identity : mapping, optional
        TO baff945c: when *config* is a science config, the identity record
        ``protocol.science_training_config`` returned alongside it (lineage,
        lineage index, and frozen-input sha256s). Carried into the task's
        metadata so the H-v2 accepted receipt can record it; ``None`` for the
        pipeline-check payload. TO 63690220: also carries a pilot/part-(g)
        payload's ``protocol.pilot_training_config`` identity record when
        *payload_label* is one of those payloads -- the two uses are
        distinguished by *payload_label*, never by the identity record's own
        shape.
    payload_label : str, optional
        TO 63690220: an explicit payload label to record instead of deriving
        one from whether *science_config_identity* is ``None``
        (``"science"``/``"pipeline-check"``). Used for the pilot/part-(g)
        payloads, whose own identity record is carried the same way a
        science identity is, but under the payload's own name
        (``"gpu-route-g"``/``"capacity-pilot"``), never mislabelled
        ``"science"``.

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
    config = copy.deepcopy(dict(config)) if config is not None else protocol.training_config()
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
    metadata: dict[str, Any] = {"route": ROUTE_NAME, "lineage": lineage}
    if science_config_identity is not None:
        metadata["payload"] = payload_label if payload_label is not None else "science"
        metadata["science_config_identity"] = {
            **dict(science_config_identity),
            "materialized_config_sha256": file_sha256(config_path),
        }
    else:
        metadata["payload"] = payload_label if payload_label is not None else "pipeline-check"
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
        metadata=metadata,
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


def _merge_config_with_overrides(effective_config: str | Path, overrides: Sequence[str]) -> Any:
    """Merge the effective config file with its exact overrides, unresolved.

    Mirrors ``experiments.toolkit.attached_activation.check_profile``'s own
    composition (``OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))``)
    so the contract check and the profile check see the same bound config.
    """

    base = OmegaConf.load(effective_config)
    return OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))


def _resolve_bound_config(merged: Any) -> dict[str, Any]:
    """Return the FULLY RESOLVED composition, for checking actual values.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``custom-resolver-required``, if resolving needs an OmegaConf
        resolver this module does not register (it never imports
        ``tpen.config``, which is where H-v2's custom resolvers, if any were
        ever added, would be registered).  H-v2's real configs use only plain
        ``${a.b.c}`` interpolations today, so this is a defensive STOP for a
        future config change, not a path exercised by the current configs.
    """

    try:
        composed = OmegaConf.to_container(merged, resolve=True)
    except UnsupportedInterpolationType as error:
        raise HV2ReceiptRefusal(
            "custom-resolver-required",
            f"resolving the bound composition needs an OmegaConf resolver this module does "
            f"not register ({error}); this module never imports tpen.config and must stop "
            f"rather than guess at one",
        ) from error
    assert isinstance(composed, dict)
    return composed


def _unresolved_bound_config(merged: Any) -> dict[str, Any]:
    """Return the bound composition with interpolations left as literal text.

    Used only to recover an artifact path's ``${run.dir}``-relative portion
    (see :func:`_run_dir_relative`): ``run.dir`` is not yet known at prepare
    time, so resolving it here would produce the wrong (or a placeholder)
    value, not a usable relative path.
    """

    unresolved = OmegaConf.to_container(merged, resolve=False)
    assert isinstance(unresolved, dict)
    return unresolved


def _run_dir_relative(value: Any, *, name: str) -> str:
    """Strip the unresolved ``${run.dir}/`` prefix from a configured path.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``unbound-artifact-path``, if ``value`` is not an unresolved
        string of exactly this shape -- a config that points an artifact
        somewhere not under the run directory is not something this module
        can safely declare a relative requirement for.
    """

    if not isinstance(value, str) or not value.startswith(_RUN_DIR_PREFIX):
        raise HV2ReceiptRefusal(
            "unbound-artifact-path", f"{name} must be an unresolved {_RUN_DIR_PREFIX!r}-relative path, got {value!r}"
        )
    return value[len(_RUN_DIR_PREFIX):]


def _derive_artifact_declarations(unresolved: Mapping[str, Any], *, stage: str) -> dict[str, str]:
    """Derive required artifact declarations from the bound, unresolved config.

    D3: a configured ``tpen.callback.InitialCheckpoint`` or
    ``tpen.callback.ParameterActivity`` makes its own output mandatory, and an
    EVAL config's trajectory records are required at their OWN configured
    path rather than a hardcoded guess.

    Parameters
    ----------
    unresolved : mapping
        The exact bound composition (effective config plus overrides),
        BEFORE interpolation resolution (see :func:`_unresolved_bound_config`).
    stage : {"train", "eval"}
        Segment stage.

    Returns
    -------
    dict
        ``{name: relative_path}`` entries to require, beyond the base
        (``resolved_config``, ``metadata``) artifacts every segment already
        declares unconditionally.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``unbound-artifact-path`` (see :func:`_run_dir_relative`), or
        code ``missing-evaluator-task`` if an EVAL config declares no
        ``"trajectory"`` evaluator task -- silently requiring nothing here
        would reintroduce the exact silently-accepted-missing-trajectory
        defect this module's trajectory check exists to refuse.
    """

    declared: dict[str, str] = {}
    for callback in unresolved.get("callbacks") or ():
        if not isinstance(callback, Mapping):
            continue
        target = callback.get("_target_")
        if target == "tpen.callback.InitialCheckpoint":
            declared["initial_checkpoint"] = _run_dir_relative(
                callback.get("output_dir"), name="InitialCheckpoint.output_dir"
            )
        elif target == "tpen.callback.ParameterActivity":
            declared["parameter_activity"] = _run_dir_relative(
                callback.get("output_path"), name="ParameterActivity.output_path"
            )
    if stage == EVAL_STAGE:
        tasks = ((unresolved.get("evaluator") or {}).get("tasks")) or ()
        trajectory_task = next(
            (entry for entry in tasks if isinstance(entry, Mapping) and entry.get("name") == "trajectory"), None
        )
        if trajectory_task is None:
            raise HV2ReceiptRefusal(
                "missing-evaluator-task", "EVAL config declares no 'trajectory' evaluator task"
            )
        relative = _run_dir_relative(trajectory_task.get("output_dir"), name="evaluator.tasks[trajectory].output_dir")
        declared["trajectory_csv"] = f"{relative}/sampled_eval_table.csv"
        declared["trajectory_metadata"] = f"{relative}/sampled_eval_table.metadata.json"
    return declared


def _reconcile_artifact_declarations(
    derived: Mapping[str, str], supplied: Mapping[str, str] | None
) -> dict[str, str]:
    """Merge a caller-supplied declaration map into the derived one, refusing a conflict.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``artifact-declaration-mismatch``, if ``supplied`` restates a
        name the bound composition already derives, with a different path.
    """

    declared = dict(derived)
    if supplied:
        for name, relative in supplied.items():
            if name in declared and declared[name] != relative:
                raise HV2ReceiptRefusal(
                    "artifact-declaration-mismatch",
                    f"caller-supplied declaration for {name!r} ({relative!r}) disagrees with the "
                    f"bound composition's own ({declared[name]!r})",
                )
            declared[name] = relative
    return declared


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
        If the plan does not carry exactly one task; code
        ``custom-resolver-required``, ``unbound-artifact-path``, or
        ``missing-evaluator-task`` from resolving and deriving artifact
        declarations (see :func:`_resolve_bound_config`,
        :func:`_derive_artifact_declarations`).
    ValueError
        If the exact bound config -- the effective config file composed with
        its own overrides and FULLY RESOLVED, the same values that will
        actually launch -- fails :func:`protocol.assert_hydrogen_contract`.
        Carried requirement 1 is checked here, on the RESOLVED composition,
        not the unresolved dict: an override supplied after the builder ran
        (for example a tampered ``system.n_particles=2``, or one expressed
        through an interpolation) is caught here, before any dispatch.
    ActivationRefusal
        If handoff translation fails; current sidecars are retired and a
        diagnostic is preserved (see ``write_attached_handoff``).
    """

    if len(plan.tasks) != 1:
        raise HV2ReceiptRefusal("multi-task-plan", "the H-v2 attached route dispatches exactly one task")
    task = plan.tasks[0]
    effective_config = task.params["config"]
    overrides = tuple(str(item) for item in overrides_by_task[task.logical_task_id])
    merged = _merge_config_with_overrides(effective_config, overrides)
    resolved = _resolve_bound_config(merged)
    protocol.assert_hydrogen_contract(resolved, evaluation=(task.stage == EVAL_STAGE))
    unresolved = _unresolved_bound_config(merged)
    artifact_declarations = _derive_artifact_declarations(unresolved, stage=task.stage)
    handoff = write_attached_handoff(
        output_dir,
        plan,
        effective_config=effective_config,
        source_config=effective_config,
        overrides_by_task=overrides_by_task,
        metadata={"route": ROUTE_NAME, "stage": task.stage},
    )
    return PreparedHV2Run(
        plan,
        handoff,
        task.logical_task_id,
        str(effective_config),
        dict(config),
        overrides,
        artifact_declarations,
        payload=task.metadata.get("payload", "pipeline-check"),
        science_config_identity=task.metadata.get("science_config_identity"),
    )


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
    enabled_modules: Sequence[str] | None = None,
    skipped_not_admitted: Sequence[str] | None = None,
    skipped_unspecified: Sequence[Mapping[str, str]] | None = None,
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
    enabled_modules, skipped_not_admitted : sequence of str, optional
        TO 63690220 amendment r3: forwarded verbatim to
        :func:`write_h_v2_accepted_receipt`.
    skipped_unspecified : sequence of mapping, optional
        TO 63690220 amendment r4: forwarded verbatim
        (:func:`protocol.unspecified_modules`).
    artifact_declarations : mapping[str, str], optional
        Additional ``{name: relative_path}`` entries, beyond the base
        artifacts and whatever D3 already derives from ``prepared``'s own
        bound composition (a configured ``InitialCheckpoint``/
        ``ParameterActivity`` output, and -- for EVAL -- the trajectory pair
        at its configured path).  Reconciled against that derived map (see
        :func:`_reconcile_artifact_declarations`): a name this function
        already derives must agree on its path, or the run refuses with
        ``artifact-declaration-mismatch``.  Later logging layers extend this
        without redefining the receipt schema.

    Returns
    -------
    HV2Activation
        Profile, dispatch, and both receipt paths.

    Raises
    ------
    ActivationRefusal
        If the profile check or the executor dispatch -- including its own
        sidecar preflight, before any subprocess is spawned -- is refused.
    HV2ReceiptRefusal, ManifestRefusal
        If the H-v2 accepted receipt refuses the run (see
        :func:`build_h_v2_accepted_receipt`), or the caller-supplied
        ``artifact_declarations`` disagrees with the derived ones.  Both the
        generic and the H-v2 failure receipts are written before the
        exception propagates, each carrying the ORIGINAL named code and
        reason.
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
    try:
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
    except ActivationRefusal as error:
        # D2: run_executor_cli's own sidecar preflight (_validate_attached_sidecars)
        # can refuse before any subprocess is spawned; that refusal must retire
        # both receipts exactly like the profile-check refusal above, not reach
        # the caller with stale receipt bytes left in place.
        _fail(error.code, error.reason, profile=profile.profile)
        raise
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
    except ManifestRefusal as error:
        # D2: write_activation_receipt already preserves error.code/reason on
        # the operational receipt itself; preserve the SAME named code/reason
        # on the H-v2 receipt too, and re-raise the original ManifestRefusal
        # unchanged (it already carries .code/.reason) instead of substituting
        # a generic "reader-failed" placeholder that erases them.
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
            code="reader-failed",
            reason=str(error),
            expected_task_ids=expected,
            profile=profile.profile,
            manifest_path=manifest_path,
        )
        raise ActivationRefusal("reader-failed", str(error)) from error
    try:
        reconciled_artifact_declarations = _reconcile_artifact_declarations(
            prepared.artifact_declarations, artifact_declarations
        )
        write_h_v2_accepted_receipt(
            manifest_path,
            h_v2_receipt_path,
            expected_task_ids=expected,
            expected_source_commit=expected_source_commit,
            stage=stage,
            inspect_run=inspect_run,
            artifact_declarations=reconciled_artifact_declarations,
            payload=prepared.payload,
            science_config_identity=prepared.science_config_identity,
            enabled_modules=enabled_modules,
            skipped_not_admitted=skipped_not_admitted,
            skipped_unspecified=skipped_unspecified,
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


#: Checkpoint step directories carry this sentinel file once the write that
#: produced them has committed (``tpen.checkpoint.artifact.COMPLETE_MARKER``,
#: mirrored here as a filesystem convention rather than importing ``tpen``:
#: ``experiments/README.md`` forbids any import of ``tpen`` outside
#: ``tpen.inspection``). A directory missing this marker, or still named
#: with the ``.tmp`` suffix, is an in-progress write and is never inventoried.
_CHECKPOINT_COMPLETE_MARKER = "COMPLETE"
_CHECKPOINT_MANIFEST_NAME = "manifest.json"
_CHECKPOINT_STEP_DIR_PREFIX = "step_"
#: Mirrored from ``tpen.checkpoint.manifest.CHECKPOINT_SCHEMA_VERSION`` as a
#: filesystem convention (``experiments/README.md`` forbids importing
#: anything from ``tpen`` here beyond ``tpen.inspection``). Only a v2
#: manifest records ``completed_updates``; a v1/legacy or malformed manifest
#: is refused, never relabelled (F1).
_CHECKPOINT_MANIFEST_SCHEMA_VERSION_V2 = 2


def _completed_updates_from_manifest(manifest_path: Path, entry: Path) -> int:
    """Return the checkpoint's scientific update coordinate from its manifest.

    F1: a checkpoint directory's own name (``step_<next_iteration>``) is the
    trainer's RESUME CURSOR, not the count of optimizer updates actually
    applied -- the two counters diverge whenever a completed iteration
    skipped its update. The manifest's ``completed_updates`` field is the
    coordinate E must select checkpoints by; the directory cursor is kept
    only to reconcile against it, never substituted for it.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``malformed-checkpoint-manifest``, if *manifest_path* is not
        readable JSON, is not schema version 2, or omits
        ``completed_updates``/``next_iteration`` (a v1/legacy manifest never
        recorded ``completed_updates`` and is refused here, never
        relabelled). Code ``checkpoint-cursor-mismatch``, if the directory's
        own cursor disagrees with the manifest's recorded ``next_iteration``.
    """

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = int(data["schema_version"])
        next_iteration = int(data["next_iteration"])
        completed_updates = int(data["completed_updates"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise HV2ReceiptRefusal(
            "malformed-checkpoint-manifest",
            f"{manifest_path} does not carry a readable v2 completed_updates/next_iteration "
            f"pair: {error}",
        ) from error
    if schema_version != _CHECKPOINT_MANIFEST_SCHEMA_VERSION_V2:
        raise HV2ReceiptRefusal(
            "malformed-checkpoint-manifest",
            f"{manifest_path} is schema_version {schema_version}, expected "
            f"{_CHECKPOINT_MANIFEST_SCHEMA_VERSION_V2} (a legacy v1 manifest never recorded "
            "completed_updates and is refused, never relabelled)",
        )
    directory_cursor = int(entry.name[len(_CHECKPOINT_STEP_DIR_PREFIX):])
    if next_iteration != directory_cursor:
        raise HV2ReceiptRefusal(
            "checkpoint-cursor-mismatch",
            f"{entry} directory cursor {directory_cursor} disagrees with manifest "
            f"next_iteration {next_iteration}",
        )
    return completed_updates


def build_checkpoint_inventory(checkpoint_root: str | Path) -> list[dict[str, Any]]:
    """Return every retained, committed checkpoint under *checkpoint_root*.

    TO 63690220 orchestrator decision 1: the pilot/part-(g) TRAIN receipt
    records the retained-checkpoint inventory (update, path, sha256 of each
    file) so item E can hash-on-read against it rather than trust-on-first-use.

    Parameters
    ----------
    checkpoint_root : str or pathlib.Path
        The ``Checkpoint``/``InitialCheckpoint`` callbacks' shared
        ``output_dir`` (``${run.dir}/checkpoints`` in these configs).

    Returns
    -------
    list of dict
        One entry per committed ``step_<N>`` directory, sorted by update
        number: ``{"update": int, "path": str, "files": {filename: sha256}}``.
        F1: ``update`` is the manifest's ``completed_updates`` -- the
        scientific update coordinate E selects checkpoints by -- not the
        directory's own resume-cursor name. A directory still suffixed
        ``.tmp``, or missing its :data:`_CHECKPOINT_COMPLETE_MARKER`/manifest,
        is an in-progress write and is never inventoried.

    Raises
    ------
    HV2ReceiptRefusal
        See :func:`_completed_updates_from_manifest`.
    """

    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    inventory: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.endswith(".tmp") or not entry.name.startswith(_CHECKPOINT_STEP_DIR_PREFIX):
            continue
        manifest_path = entry / _CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file() or not (entry / _CHECKPOINT_COMPLETE_MARKER).is_file():
            continue
        update = _completed_updates_from_manifest(manifest_path, entry)
        files = {
            str(file_path.relative_to(entry)): file_sha256(file_path)
            for file_path in sorted(entry.rglob("*"))
            if file_path.is_file()
        }
        inventory.append({"update": update, "path": str(entry), "files": files})
    inventory.sort(key=lambda entry: entry["update"])
    return inventory


def build_h_v2_accepted_receipt(
    manifest_path: str | Path,
    *,
    expected_task_ids: Sequence[str],
    expected_source_commit: str,
    stage: str = TRAIN_STAGE,
    inspect_run: Any = None,
    artifact_declarations: Mapping[str, str] | None = None,
    payload: str = "pipeline-check",
    science_config_identity: Mapping[str, Any] | None = None,
    enabled_modules: Sequence[str] | None = None,
    skipped_not_admitted: Sequence[str] | None = None,
    skipped_unspecified: Sequence[Mapping[str, str]] | None = None,
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
    payload : {"science", "pipeline-check"}, default="pipeline-check"
        TO baff945c: which payload this TRAIN segment dispatched. Recorded
        verbatim on the receipt.
    science_config_identity : mapping, optional
        Required when ``payload == "science"``: the identity record
        returned alongside the config by ``protocol.science_training_config``
        plus ``materialized_config_sha256``
        (:data:`SCIENCE_CONFIG_IDENTITY_KEYS`). A science receipt missing any
        of these fields refuses rather than being written incomplete.
    enabled_modules, skipped_not_admitted : sequence of str, optional
        TO 63690220 amendment r3: which A1 training-side module targets
        (:data:`protocol.A1_TRAINING_MODULE_CANDIDATES`) the runtime recovery
        profile admitted versus refused. Skipping is never silent: both
        default to ``[]`` rather than being omitted.
    skipped_unspecified : sequence of mapping, optional
        TO 63690220 amendment r4: modules never probed or appended because
        their configuration is not yet specified upstream
        (:func:`protocol.unspecified_modules`), each as
        ``{"target": ..., "reason": ...}``. Defaults to ``[]``.

    Returns
    -------
    dict
        The ``tpen-experiments/h-v2-accepted-receipt-v1`` payload, keyed per
        accepted segment by ``(task_id, attempt_id, checkpoint_dir)``.

    Raises
    ------
    HV2ReceiptRefusal
        Code ``science-config-identity-missing``, if ``payload == "science"``
        and ``science_config_identity`` omits any required field.
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
    if payload == "science":
        missing = SCIENCE_CONFIG_IDENTITY_KEYS - set(science_config_identity or {})
        if missing:
            raise HV2ReceiptRefusal(
                "science-config-identity-missing",
                f"payload=science receipt is missing required science config identity fields: "
                f"{sorted(missing)!r}",
            )
    elif payload in protocol.PILOT_PAYLOADS:
        # K1: a pilot/part-(g) receipt must carry its own complete identity
        # record too (coverage gap: previously checked for payload=="science"
        # only, so a dropped/corrupted payload_overlay_sha256 anywhere along
        # protocol -> build_h_v2_train_plan -> this receipt went unnoticed).
        missing = PILOT_CONFIG_IDENTITY_KEYS - set(science_config_identity or {})
        if missing:
            raise HV2ReceiptRefusal(
                "pilot-config-identity-missing",
                f"payload={payload!r} receipt is missing required pilot config identity fields: "
                f"{sorted(missing)!r}",
            )
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

    # TO 63690220 orchestrator decision 1: for TRAIN, every segment's
    # checkpoint_dir names the LATEST committed checkpoint step under the
    # callbacks' shared output_dir; that directory's parent is the full
    # checkpoints root, so the inventory is derived here rather than asking
    # the caller to resolve a run-time ${run.dir} path it does not itself know.
    checkpoint_inventory: list[dict[str, Any]] = []
    if stage == TRAIN_STAGE and segments_payload:
        checkpoint_root = Path(segments_payload[0]["checkpoint_dir"]).parent
        checkpoint_inventory = build_checkpoint_inventory(checkpoint_root)

    by_task: dict[str, list[str]] = {}
    for segment_payload in segments_payload:
        by_task.setdefault(segment_payload["task_id"], []).append(segment_payload["resolved_config_sha256"])
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
        "payload": payload,
        "science_config_identity": dict(science_config_identity) if science_config_identity is not None else None,
        "checkpoint_inventory": checkpoint_inventory,
        "enabled_modules": list(enabled_modules) if enabled_modules else [],
        "skipped_not_admitted": list(skipped_not_admitted) if skipped_not_admitted else [],
        "skipped_unspecified": [dict(entry) for entry in skipped_unspecified] if skipped_unspecified else [],
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
    "PILOT_CONFIG_IDENTITY_KEYS",
    "PreparedHV2Run",
    "ROUTE_NAME",
    "SCIENCE_CONFIG_IDENTITY_KEYS",
    "TRAIN_STAGE",
    "activate_h_v2_run",
    "build_checkpoint_inventory",
    "build_h_v2_accepted_receipt",
    "build_h_v2_eval_plan",
    "build_h_v2_train_plan",
    "prepare_h_v2_run",
    "assert_candidate_instantiates",
    "probe_admitted_a1_modules",
    "write_h_v2_accepted_receipt",
]
