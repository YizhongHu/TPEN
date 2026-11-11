"""Frozen H-v2 pipeline configuration helpers and identity checks."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.toolkit.attached_activation import file_sha256

ROOT = Path(__file__).parent
TRAINING_SEED = 910001
SAMPLER_SEED = 910002
EVALUATION_SEEDS = (910003, 910004)

#: A1/TO baff945c: scale-only overrides onto the frozen pipeline-check
#: training config, applied by explicit dotted key path (never a blind
#: ``OmegaConf.merge``, which replaces ``model.layers`` wholesale instead of
#: touching one nested leaf). The overlay file's key set must equal this
#: allowlist exactly.
SCIENCE_SCALE_OVERLAY_PATH = ROOT / "configs" / "science_scale_overlay.yaml"
SCIENCE_SCALE_OVERLAY_KEYS = frozenset(
    {
        "runtime.device",
        "sampler.n_walkers",
        "trainer.max_steps",
        "model.embedding.out_channels",
        "model.embedding.hidden_channels",
        "model.layers.0.mixing.channels",
        "model.layers.0.path_aggregation.channels",
        "model.readout.channels",
    }
)

#: Source of per-lineage science seeds (A1: "no seed literal outside CI").
CONTRACT_INPUTS_PATH = ROOT / "contract_inputs.json"

#: TO 63690220 (pilot and part (g) TRAIN payloads): per-payload lineage
#: label, payload-overlay path, and CI non-science seed namespace. Each
#: payload refuses any other lineage or seed (acceptance-contract amendment
#: r2 P1/P2).
PILOT_PAYLOADS: dict[str, dict[str, Any]] = {
    "gpu-route-g": {
        "lineage": "G",
        "overlay_path": ROOT / "configs" / "gpu_route_g_overlay.yaml",
        "seed_namespace": "pipeline_gpu_smoke_g",
    },
    "capacity-pilot": {
        "lineage": "P",
        "overlay_path": ROOT / "configs" / "capacity_pilot_overlay.yaml",
        "seed_namespace": "capacity_pilot",
    },
}

#: Allowlist for both pilot payload overlays (A1/TO 63690220):
#: ``trainer.max_steps`` and the A1-cadence keys of the two A1 training-side
#: modules that are already present leaves in ``train.yaml`` (DataIntegrity
#: at ``callbacks.4``, GradientStats at ``callbacks.5``). Every other A1
#: training-side module this payload enables is appended, never set here --
#: ``_set_by_dotted_path`` only ever touches a leaf that already exists.
PILOT_PAYLOAD_OVERLAY_KEYS = frozenset(
    {
        "trainer.max_steps",
        "trainer.log_every_n_steps",
        "callbacks.4.every_n_steps",
        "callbacks.5.every_n_steps",
    }
)

#: A1 cadence value shared by every pilot/part-(g) measurement callback:
#: every update (acceptance-contract: "log_every_n_steps 1; DataIntegrity,
#: GradientStats every 1").
PILOT_A1_CADENCE = 1

HEALTH_CALLBACK_CADENCE = 1
TRAINING_BATCH_RECORDER_CADENCE = 50
TRAINING_HISTORY_PARTITION_SIZE = 1000

#: TO 63690220 amendment r3: every "A1 training-side logging module"
#: candidate the pilot tries to enable at A1 cadence. Admission is decided
#: per candidate, never as a block, by probing the runtime recovery profile
#: (:func:`experiments.toolkit.attached_activation.check_profile`) -- never
#: assumed from the branch. ``DataIntegrity``/``GradientStats`` are already
#: present leaves in ``train.yaml`` (cadence set by the payload overlay
#: above); the rest are appended fresh by :func:`append_measurement_callbacks`.
#: The study slice freezes the remaining diagnostic candidates below. Grouped
#: callbacks receive paths from :func:`module_norm_groups`, and attached.py
#: checks those paths against the instantiated model before dispatch.
A1_TRAINING_MODULE_CANDIDATES: tuple[str, ...] = (
    "tpen.callback.DataIntegrity",
    "tpen.callback.GradientStats",
    "tpen.callback.SamplerHealth",
    "tpen.callback.FactorScalars",
    "tpen.callback.ModuleNorms",
    "tpen.callback.OptimizerState",
    "tpen.callback.RawTrainingBatchRecorder",
    "tpen.callback.InitialCheckpoint",
    "tpen.callback.ParameterActivity",
    "tpen.callback.ElectronNucleusDistanceQuantiles",
)

#: No H-v2 training modules are unspecified after this slice.
UNSPECIFIED_MODULES: dict[str, str] = {}

#: Frozen H-v2 module-group paths. ``attached.py`` resolves these against the
#: instantiated model before enabling grouped callbacks.
MODULE_NORM_GROUPS: dict[str, tuple[str, ...]] = {
    "embedding": ("embedding",),
    "readout": ("readout",),
    "factors": ("factors",),
}

#: Candidates already present as leaves in the base ``train.yaml`` callbacks
#: list; never appended again by :func:`append_candidate_module`.
_A1_CANDIDATES_PRESENT_IN_BASE_CONFIG = frozenset(
    {"tpen.callback.DataIntegrity", "tpen.callback.GradientStats"}
)

#: Design doc A1's frozen quantile levels for
#: ``tpen.callback.ElectronNucleusDistanceQuantiles``, taken from this one
#: protocol constant (amendment r4 rule 1), never a second literal copy.
ENDQ_A1_QUANTILES: tuple[float, ...] = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999)


def unspecified_modules() -> list[dict[str, str]]:
    """Return :data:`UNSPECIFIED_MODULES` as the receipt's ``skipped_unspecified`` shape."""

    return [{"target": target, "reason": reason} for target, reason in UNSPECIFIED_MODULES.items()]

#: Contract-frozen nuclear cusp law literals (``admission.pipeline_check.nuclear_law_note``).
CONTRACT_NUCLEAR_LAW = {
    "_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw",
    "curvature_coefficient": 0.01,
    "curvature_range": 1.0,
    "trainable": True,
}

#: Frozen amplitude-risk study thresholds (``measurement.amplitude_risk.thresholds``
#: in ``contract_inputs.json``, decided by note ``amplitude-risk-call-20261106``).
#: Descriptive only: this list gates nothing on its own, it is the single value
#: every configured `tpen.callback.AmplitudeRiskFraction` instance in an H-v2
#: config must match, checked by `assert_hydrogen_contract`.
AMPLITUDE_RISK_THRESHOLDS = (-3, -4, -5, -7, -10)
AMPLITUDE_RISK_PRIMARY_THRESHOLD = -5
AMPLITUDE_RISK_LABELS = {
    "-3": "resolved sensitivity",
    "-4": "resolved sensitivity",
    "-5": "primary descriptive scale",
    "-7": (
        "marginal resolution at the spike's evaluation scale (4096 x 64 rows "
        "per stream, 17 of 20 streams resolved); unmeasured at the contract's "
        "4096 x 256"
    ),
    "-10": "rare or unresolved; a zero is not a healthy zero",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one frozen configuration without resolving Hydra interpolations."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def training_config() -> dict[str, Any]:
    """Return a fresh copy of the frozen short-route training config."""

    return copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml"))


def _set_by_dotted_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set one leaf of *config* addressed by a ``.``-separated path.

    A purely-numeric segment indexes a list (``model.layers.0...``); any
    other segment indexes a mapping. This never creates a missing
    intermediate node -- the overlay only ever touches leaves that already
    exist in the frozen base config.
    """

    parts = dotted.split(".")
    node: Any = config
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def _science_seed_roster() -> dict[str, list[int]]:
    """Read the frozen per-lineage training seeds from ``contract_inputs.json``.

    Returns
    -------
    dict
        ``measurement.seed_roster``'s ``training_initialization_seeds`` and
        ``training_sampler_seeds`` lists -- the single frozen source; no
        seed literal is duplicated here.
    """

    payload = json.loads(CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    roster = payload["measurement"]["seed_roster"]
    return {
        "training_initialization_seeds": roster["training_initialization_seeds"],
        "training_sampler_seeds": roster["training_sampler_seeds"],
    }


def science_training_config(lineage: str, lineage_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one lineage's science TRAIN config, plus its config identity.

    Parameters
    ----------
    lineage : str
        Opaque lineage identity (for example ``"L1"``); recorded in the
        identity, never used here to decide which lineages are approved --
        that roster stays owned by ``launch_config.LINEAGES``, checked by
        the caller before this function is ever reached.
    lineage_index : int
        0-based index into ``contract_inputs.json``'s seed roster
        (``launch_config.LINEAGES.index(lineage)``).

    Returns
    -------
    tuple of dict, dict
        The resolved science training config (base ``train.yaml`` plus the
        scale overlay plus the lineage's frozen seeds), and a science-config
        identity record (lineage, lineage index, and the sha256 of every
        frozen input file this config was built from).

    Raises
    ------
    ValueError
        If the overlay file's key set has drifted from
        :data:`SCIENCE_SCALE_OVERLAY_KEYS`.
    """

    config = training_config()
    overlay = load_yaml(SCIENCE_SCALE_OVERLAY_PATH)
    if set(overlay) != SCIENCE_SCALE_OVERLAY_KEYS:
        raise ValueError(
            f"science scale overlay key set {set(overlay)!r} does not match the allowlist "
            f"{set(SCIENCE_SCALE_OVERLAY_KEYS)!r}"
        )
    for dotted, value in overlay.items():
        _set_by_dotted_path(config, dotted, value)
    seeds = _science_seed_roster()
    config["runtime"]["seed"] = seeds["training_initialization_seeds"][lineage_index]
    config["sampler"]["seed"] = seeds["training_sampler_seeds"][lineage_index]
    identity = {
        "lineage": lineage,
        "lineage_index": lineage_index,
        "overlay_sha256": file_sha256(SCIENCE_SCALE_OVERLAY_PATH),
        "base_train_sha256": file_sha256(ROOT / "configs" / "train.yaml"),
        "contract_inputs_sha256": file_sha256(CONTRACT_INPUTS_PATH),
    }
    return config, identity


def _non_science_seed_namespace(namespace: str) -> dict[str, list[int]]:
    """Read one frozen non-science seed namespace from ``contract_inputs.json``.

    Raises
    ------
    KeyError
        If *namespace* is not declared in
        ``measurement.seed_roster.non_science_seed_namespaces``.
    """

    payload = json.loads(CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    roster = payload["measurement"]["seed_roster"]["non_science_seed_namespaces"]
    return roster[namespace]


def _build_candidate_callback(target: str) -> dict[str, Any]:
    """Return one A1 training-side module's candidate callback entry.

    Raises
    ------
    ValueError
        If *target* is not one of :data:`A1_TRAINING_MODULE_CANDIDATES`.
    """

    if target in {"tpen.callback.ModuleNorms", "tpen.callback.OptimizerState"}:
        return {
            "_target_": target,
            "every_n_steps": HEALTH_CALLBACK_CADENCE,
            "groups": {name: list(paths) for name, paths in module_norm_groups().items()},
        }
    if target == "tpen.callback.SamplerHealth":
        return {"_target_": target, "every_n_steps": HEALTH_CALLBACK_CADENCE, "fail_fast": False}
    if target == "tpen.callback.FactorScalars":
        return {"_target_": target, "every_n_steps": HEALTH_CALLBACK_CADENCE, "fail_fast": False}
    if target == "tpen.callback.RawTrainingBatchRecorder":
        return {
            "_target_": target,
            "path": "${run.dir}/training_history",
            "every_n_steps": TRAINING_BATCH_RECORDER_CADENCE,
            "partition_size_steps": TRAINING_HISTORY_PARTITION_SIZE,
        }
    if target == "tpen.callback.InitialCheckpoint":
        # No cadence field: it fires exactly once, at the pre-sampling
        # zero-update boundary.
        return {"_target_": target, "output_dir": "${run.dir}/checkpoints"}
    if target == "tpen.callback.ParameterActivity":
        # No cadence field: it fires exactly once, on the first applied update.
        return {"_target_": target, "output_path": "${run.dir}/parameter_activity.json"}
    if target == "tpen.callback.ElectronNucleusDistanceQuantiles":
        # Design doc A1 exactly: no output_path -- the real constructor
        # (``ElectronNucleusDistanceQuantiles(*, quantiles, **kwargs)``) takes
        # none, and a config key that signature does not accept would pass
        # this module's own syntax-only profile probe and fail only at real
        # instantiation (amendment r4).
        return {
            "_target_": target,
            "every_n_steps": PILOT_A1_CADENCE,
            "quantiles": list(ENDQ_A1_QUANTILES),
        }
    raise ValueError(f"no candidate-callback builder for A1 training-side module {target!r}")


def module_norm_groups(config: Mapping[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    """Return frozen H-v2 groups for the configured model's module paths."""

    model = (config or training_config()).get("model", {})
    groups = dict(MODULE_NORM_GROUPS)
    for index in range(len(model.get("layers", ()) or ())):
        groups[f"layer_{index}_mixing"] = (f"stack.layers.{index}.mixing",)
        groups[f"layer_{index}_path_aggregation"] = (f"stack.layers.{index}.path_aggregation",)
    return groups


def candidate_modules_to_append() -> tuple[str, ...]:
    """Return the A1 training-side module candidates not already in ``train.yaml``."""

    return tuple(
        target
        for target in A1_TRAINING_MODULE_CANDIDATES
        if target not in _A1_CANDIDATES_PRESENT_IN_BASE_CONFIG
    )


def append_candidate_module(config: dict[str, Any], target: str) -> None:
    """Append one candidate module's callback entry onto *config* in place."""

    entry = _build_candidate_callback(target)
    checkpoint_index = next(
        (index for index, callback in enumerate(config["callbacks"])
         if callback.get("_target_") == "tpen.callback.Checkpoint"),
        len(config["callbacks"]),
    )
    config["callbacks"].insert(checkpoint_index, entry)


def remove_candidate_module(config: dict[str, Any], target: str) -> None:
    """Remove a previously appended candidate module's entry from *config* in place."""

    config["callbacks"] = [
        callback for callback in config["callbacks"] if callback.get("_target_") != target
    ]


def reconcile_enabled_modules(config: dict[str, Any]) -> list[str]:
    """Return every :data:`A1_TRAINING_MODULE_CANDIDATES` target actually in *config*.

    F4: the TRAIN receipt's ``enabled_modules`` must be the complete,
    disjoint partition of the A1 training-side candidate set that actually
    ran -- including the base ``DataIntegrity``/``GradientStats`` leaves
    already present in ``train.yaml``, which :func:`candidate_modules_to_append`
    deliberately excludes from the probed candidate set (they are never
    appended or removed). Reconciling against the final materialized
    config's own callbacks, rather than trusting the probe's own
    bookkeeping, means a candidate silently dropped or duplicated between
    probing and dispatch cannot drift the receipt from what actually ran.

    Parameters
    ----------
    config : mapping
        The final materialized config (after probing), as actually written
        to the task's effective config file.

    Returns
    -------
    list of str
        Candidates in :data:`A1_TRAINING_MODULE_CANDIDATES` order whose
        ``_target_`` is present in ``config["callbacks"]``.
    """

    present = {
        callback.get("_target_")
        for callback in config.get("callbacks") or ()
        if isinstance(callback, Mapping)
    }
    return [target for target in A1_TRAINING_MODULE_CANDIDATES if target in present]


def append_measurement_callbacks(config: dict[str, Any]) -> None:
    """Append the pilot's own timing callbacks onto *config* in place.

    TO 63690220: ``RunTiming`` and ``TrainStepTiming`` (A1 cadence,
    ``accelerator_synchronize: false``), appended by this one protocol-owned
    helper -- distinct from the dotted-path overlay mechanism, which can only
    set a leaf that already exists in ``train.yaml``.
    """

    config["callbacks"].append({"_target_": "tpen.callback.RunTiming"})
    config["callbacks"].append(
        {
            "_target_": "tpen.callback.TrainStepTiming",
            "every_n_steps": PILOT_A1_CADENCE,
            "rolling_window": 20,
            "accelerator_synchronize": False,
        }
    )


def pilot_training_config(payload: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one pilot/part-(g) payload's TRAIN config, plus its identity.

    Builds ``train.yaml`` plus the UNCHANGED ``science_scale_overlay.yaml``
    (TO baff945c) plus *payload*'s own overlay (allowlisted by
    :data:`PILOT_PAYLOAD_OVERLAY_KEYS`), then appends
    :func:`append_measurement_callbacks`'s timing pair. It does NOT append
    any :data:`A1_TRAINING_MODULE_CANDIDATES` entry: the caller (which alone
    has a science interpreter to probe the runtime recovery profile) owns
    deciding which candidates are admitted, via
    :func:`append_candidate_module`/:func:`remove_candidate_module`.

    Parameters
    ----------
    payload : {"gpu-route-g", "capacity-pilot"}

    Returns
    -------
    tuple of dict, dict
        The resolved TRAIN config (before any candidate module is appended),
        and an identity record (payload, lineage, seed namespace, and the
        sha256 of every frozen input file this config was built from).

    Raises
    ------
    ValueError
        If *payload* is not a declared pilot payload, or either overlay's key
        set has drifted from its allowlist.
    """

    if payload not in PILOT_PAYLOADS:
        raise ValueError(f"payload must be one of {sorted(PILOT_PAYLOADS)!r}, got {payload!r}")
    spec = PILOT_PAYLOADS[payload]

    config = training_config()
    scale_overlay = load_yaml(SCIENCE_SCALE_OVERLAY_PATH)
    if set(scale_overlay) != SCIENCE_SCALE_OVERLAY_KEYS:
        raise ValueError(
            f"science scale overlay key set {set(scale_overlay)!r} does not match the allowlist "
            f"{set(SCIENCE_SCALE_OVERLAY_KEYS)!r}"
        )
    for dotted, value in scale_overlay.items():
        _set_by_dotted_path(config, dotted, value)

    payload_overlay = load_yaml(spec["overlay_path"])
    if set(payload_overlay) != PILOT_PAYLOAD_OVERLAY_KEYS:
        raise ValueError(
            f"{payload} overlay key set {set(payload_overlay)!r} does not match the allowlist "
            f"{set(PILOT_PAYLOAD_OVERLAY_KEYS)!r}"
        )
    for dotted, value in payload_overlay.items():
        _set_by_dotted_path(config, dotted, value)

    seeds = _non_science_seed_namespace(spec["seed_namespace"])
    config["runtime"]["seed"] = seeds["training"][0]
    config["sampler"]["seed"] = seeds["sampler"][0]

    append_measurement_callbacks(config)

    identity = {
        "payload": payload,
        "lineage": spec["lineage"],
        "seed_namespace": spec["seed_namespace"],
        "scale_overlay_sha256": file_sha256(SCIENCE_SCALE_OVERLAY_PATH),
        "payload_overlay_sha256": file_sha256(spec["overlay_path"]),
        "base_train_sha256": file_sha256(ROOT / "configs" / "train.yaml"),
        "contract_inputs_sha256": file_sha256(CONTRACT_INPUTS_PATH),
    }
    return config, identity


def evaluation_config(checkpoint: Path, *, stream: int = 0, root: Path | None = None) -> dict[str, Any]:
    """Return a fresh model-only evaluation config for one stream."""

    if stream not in (0, 1):
        raise ValueError("pipeline evaluation stream must be 0 or 1")
    config = load_yaml(ROOT / "configs" / "eval.yaml")
    config["load"]["path"] = str(checkpoint)
    config["runtime"]["seed"] = EVALUATION_SEEDS[stream]
    config["run"]["run_id"] = f"h-v2-eval-stream{stream}"
    if root is not None:
        config["run"]["root"] = str(root)
    return config


#: TO 8a5e926f (item E, design target section A2): the evaluation-side scale
#: overlay, applied onto ``eval.yaml`` the same explicit dotted-path way
#: :func:`science_training_config` applies :data:`SCIENCE_SCALE_OVERLAY_PATH`
#: onto ``train.yaml``. The model architecture keys here MUST equal
#: :data:`SCIENCE_SCALE_OVERLAY_KEYS`'s own values: this config restores a
#: checkpoint trained at those widths, so a drifted channel count here would
#: fail checkpoint restore, not merely evaluate a different model silently.
EVAL_SCALE_OVERLAY_PATH = ROOT / "configs" / "eval_scale_overlay.yaml"
EVAL_SCALE_OVERLAY_KEYS = frozenset(
    {
        "runtime.device",
        "sampler.n_walkers",
        "sampler.burn_in",
        "evaluator.tasks.0.generator.n_draws",
        "evaluator.tasks.0.summaries.0.max_samples",
        "model.embedding.out_channels",
        "model.embedding.hidden_channels",
        "model.layers.0.mixing.channels",
        "model.layers.0.path_aggregation.channels",
        "model.readout.channels",
    }
)

#: Design target D8: the science-scale evaluation config must never carry a
#: ``reference_energy`` key anywhere -- the hydrogen exact ground state is an
#: instrument control, never a training or evaluation input.
EVAL_FORBIDDEN_KEYS: tuple[str, ...] = ("reference_energy",)

#: ``tpen.nn.readout.pfaffian.PfaffianReadout``'s own default floor
#: (amendment r4 E5), restated here as the readout-trajectory generator's
#: expected ``readout_eps`` -- the generator's own internal floor check
#: (``ReadoutTrajectoryMCMCGenerator.generate``) refuses at runtime if the
#: RESTORED model's actual eps disagrees. F1 (review r1): this constant is
#: never written into a TRAIN or EVAL config's ``model.readout.eps`` --
#: doing so would change the model mapping core hashes for restore identity,
#: so an earlier TRAIN checkpoint (whose receipt's authenticated mapping has
#: no explicit ``eps``) would fail to restore against a later EVAL config
#: that added one, even though the value is numerically identical to the
#: class's own constructor default. The model mapping used for restore is
#: always the TRAIN receipt's own, untouched; only the generator's
#: *expected* floor is supplied here.
READOUT_EPS = 1.0e-12

#: Orchestrator decision 2(a) (TO 8a5e926f): enabled for every eval payload
#: (science, pilot, part g), so phase B measures its added cost.
READOUT_TRAJECTORY_GENERATOR_TARGET = (
    "tpen.evaluation.generators.readout_trajectory.ReadoutTrajectoryMCMCGenerator"
)


def _assert_no_forbidden_keys(node: Any, *, path: str = "") -> None:
    """Recursively refuse any :data:`EVAL_FORBIDDEN_KEYS` leaf anywhere in *node*."""

    if isinstance(node, Mapping):
        for key, value in node.items():
            dotted = f"{path}.{key}" if path else str(key)
            if key in EVAL_FORBIDDEN_KEYS:
                raise ValueError(f"science evaluation config carries forbidden key {dotted!r} (design target D8)")
            _assert_no_forbidden_keys(value, path=dotted)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _assert_no_forbidden_keys(value, path=f"{path}.{index}")


def enable_readout_trajectory_generator(config: dict[str, Any]) -> None:
    """Append a readout-trajectory evaluator task onto *config* in place.

    F1 (review r1, fix): the generator's ``readout_eps`` is set from the
    frozen :data:`READOUT_EPS` constant, but ``config["model"]["readout"]``
    is never touched here -- the model mapping stays exactly whatever the
    TRAIN receipt's authenticated config carries (restore identity depends
    on it matching bit-for-bit), and the generator's own runtime floor check
    (``ReadoutTrajectoryMCMCGenerator.generate``) is what actually confirms
    the restored model's real eps agrees with the expected floor supplied
    here, not a config-level mutation performed in advance.
    """

    base_task = config["evaluator"]["tasks"][0]
    readout_task = copy.deepcopy(base_task)
    readout_task["name"] = "readout_trajectory"
    readout_task["namespace"] = "readout_trajectory"
    readout_task["output_dir"] = "${run.dir}/readout_trajectory"
    readout_task["generator"] = {
        **base_task["generator"],
        "_target_": READOUT_TRAJECTORY_GENERATOR_TARGET,
        "readout_eps": READOUT_EPS,
        "replay_chunk_size": base_task["generator"]["chunk_size"],
    }
    config["evaluator"]["tasks"].append(readout_task)


def science_evaluation_config(
    checkpoint: Path,
    *,
    seed: int,
    root: Path | None = None,
    enable_readout_trajectory: bool = True,
) -> dict[str, Any]:
    """Return one science-scale fresh-evaluation config (design target A2).

    Builds ``eval.yaml`` plus :data:`EVAL_SCALE_OVERLAY_PATH` (model
    architecture and sampler scale matching :func:`science_training_config`'s
    own overlay), restores *checkpoint*, and sets the frozen per-generation
    *seed* -- never invented here; the caller computes it from
    :func:`science_eval_seed`, :func:`pilot_eval_seed`, or
    :func:`gpu_route_g_eval_seed`.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Absolute checkpoint directory. Never globbed for: the caller
        resolves it from a verified checkpoint-inventory entry
        (:mod:`eval_job`).
    seed : int
        The frozen seed for this exact generation.
    root : pathlib.Path, optional
        Overrides ``run.root``; the caller sets ``run.run_id`` separately
        (mirrors :func:`attached.build_h_v2_eval_plan`).
    enable_readout_trajectory : bool, default=True
        Whether to append the readout-trajectory generator task
        (:func:`enable_readout_trajectory_generator`). Orchestrator decision
        2(a) enables it for every eval payload; a caller that probed the
        class absent on its own base passes ``False`` and records the
        deviation instead of silently omitting it.

    Raises
    ------
    ValueError
        If the overlay's key set has drifted from :data:`EVAL_SCALE_OVERLAY_KEYS`,
        or the resolved config carries a forbidden key (:data:`EVAL_FORBIDDEN_KEYS`).
    """

    config = load_yaml(ROOT / "configs" / "eval.yaml")
    overlay = load_yaml(EVAL_SCALE_OVERLAY_PATH)
    if set(overlay) != EVAL_SCALE_OVERLAY_KEYS:
        raise ValueError(
            f"eval scale overlay key set {set(overlay)!r} does not match the allowlist "
            f"{set(EVAL_SCALE_OVERLAY_KEYS)!r}"
        )
    for dotted, value in overlay.items():
        _set_by_dotted_path(config, dotted, value)
    config["load"]["path"] = str(checkpoint)
    config["runtime"]["seed"] = seed
    if root is not None:
        config["run"]["root"] = str(root)
    if enable_readout_trajectory:
        enable_readout_trajectory_generator(config)
    _assert_no_forbidden_keys(config)
    return config


#: D8: science eval seeds must be READ from ``contract_inputs.json``'s own
#: ``measurement.seed_roster.evaluation_stream_seed_formula`` field at
#: runtime, never a hard-coded literal with no CI tie (the way
#: ``pilot_eval_seed``/``gpu_route_g_eval_seed`` are already checked against
#: their own roster fields) -- a drifted formula must change this function's
#: output, not be silently ignored.
_SCIENCE_EVAL_SEED_FORMULA_PATTERN = re.compile(
    r"(\d+)\s*\+\s*(\d+)\*lineage_index\s*\+\s*(\d+)\*evaluated_checkpoint_ordinal\s*\+\s*stream_index"
)


def science_eval_seed(lineage_index: int, checkpoint_ordinal: int, stream_index: int) -> int:
    """Return one science evaluation generation's seed, read from CI (design target C2).

    Raises
    ------
    ValueError
        If ``contract_inputs.json``'s ``evaluation_stream_seed_formula`` does
        not match the expected ``base + a*lineage_index + b*evaluated_checkpoint_ordinal
        + stream_index`` shape.
    """

    payload = json.loads(CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    formula = payload["measurement"]["seed_roster"]["evaluation_stream_seed_formula"]
    match = _SCIENCE_EVAL_SEED_FORMULA_PATTERN.search(formula)
    if match is None:
        raise ValueError(f"evaluation_stream_seed_formula {formula!r} does not match the expected shape")
    base, lineage_coeff, ordinal_coeff = (int(group) for group in match.groups())
    return base + lineage_coeff * lineage_index + ordinal_coeff * checkpoint_ordinal + stream_index


def science_evaluated_checkpoint_updates() -> tuple[int, ...]:
    """Return the 27 contract-frozen evaluated-checkpoint update counts, in order."""

    payload = json.loads(CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    return tuple(payload["measurement"]["evaluated_checkpoints"]["updates"])


def pilot_eval_seed(ordinal: int, stream: int) -> int:
    """Return one pilot evaluation generation's seed (amendment r3/r4 E1/E2).

    Formula ``996000 + 10*ordinal + stream``; every value is read-checked
    against ``contract_inputs.json``'s frozen ``capacity_pilot.evaluation``
    roster -- never a seed literal outside it.

    Raises
    ------
    ValueError
        If the computed seed is not in the CI roster.
    """

    seed = 996000 + 10 * ordinal + stream
    roster = _non_science_seed_namespace("capacity_pilot")["evaluation"]
    if seed not in roster:
        raise ValueError(
            f"pilot evaluation seed {seed} (ordinal={ordinal}, stream={stream}) is not in the CI roster"
        )
    return seed


def gpu_route_g_eval_seed() -> int:
    """Return part (g)'s single fresh-evaluation seed (999004, CI frozen)."""

    namespace = _non_science_seed_namespace("pipeline_gpu_smoke_g")
    (seed,) = namespace["evaluation"]
    return seed


#: Pilot two-stage grid (pilot-plan-r3-20261110 (a), amendment r3/r4 E1):
#: stage 1 is stream 0 at every checkpoint ordinal 0..20 (updates 0..20000).
PILOT_STAGE1_ORDINALS: tuple[int, ...] = tuple(range(0, 21))
PILOT_STAGE1_STREAM = 0


def pilot_stage1_grid() -> tuple[tuple[int, int], ...]:
    """Return stage 1's ``(ordinal, stream)`` pairs: stream 0 at ordinals 0..20."""

    return tuple((ordinal, PILOT_STAGE1_STREAM) for ordinal in PILOT_STAGE1_ORDINALS)


def pilot_stage2_window(c1: int) -> tuple[int, ...]:
    """Return stage 2's 5-checkpoint stream-1 window around *c1* (r3 (a)).

    ``lo = min(max(c1 - 2, 1), 16)``; the window is ``lo..lo+4`` inclusive.

    Raises
    ------
    ValueError
        If *c1* is outside ``3..19``.
    """

    if not 3 <= c1 <= 19:
        raise ValueError(f"c1 must be in 3..19, got {c1}")
    lo = min(max(c1 - 2, 1), 16)
    return tuple(range(lo, lo + 5))


def pilot_stage2_grid(c1: int, *, core: bool = False) -> tuple[tuple[int, int], ...]:
    """Return stage 2's ``(ordinal, stream)`` pairs, given SEL's basin ordinal *c1*.

    Stream 1 at the 5-checkpoint window around *c1* (:func:`pilot_stage2_window`);
    streams 2 and 3 at *c1*; streams 1-3 at ordinals 0 and 20. ``core``
    (``--pilot-stage 2-core``) drops ONLY ordinal 20's three entries -- "the
    same as stage 2 without ordinal 20" (amendment r3/r4 E1) -- never
    ordinal 0's. A pair already covered by stage 1 (any ordinal, stream 0)
    never appears here; a pair this function would otherwise repeat (for
    example *c1* already inside its own window) appears exactly once.
    """

    pairs: list[tuple[int, int]] = [(ordinal, 1) for ordinal in pilot_stage2_window(c1)]
    pairs.append((c1, 2))
    pairs.append((c1, 3))
    edge_ordinals = (0,) if core else (0, 20)
    for ordinal in edge_ordinals:
        for stream in (1, 2, 3):
            pairs.append((ordinal, stream))
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return tuple(deduped)


def _nearest_rank_p999(deviations: Sequence[float]) -> float:
    """Return the order statistic at index ``ceil(0.999*N)`` (1-based), no interpolation.

    Shared by :func:`w_statistic` (one stream's own deviations) and
    :func:`pooled_w_statistic` (several streams' deviations pooled after
    each is centered about its own median) -- the nearest-rank step itself
    never re-centers.

    Raises
    ------
    ValueError
        If *deviations* is empty.
    """

    values = sorted(deviations)
    n = len(values)
    if n == 0:
        raise ValueError("W requires at least one local energy")
    index = min(math.ceil(0.999 * n), n) - 1
    return values[index]


def _deviations_about_own_median(local_energies: Sequence[float]) -> list[float]:
    values = sorted(local_energies)
    n = len(values)
    if n == 0:
        raise ValueError("W requires at least one local energy")
    median = values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    return [abs(value - median) for value in local_energies]


def w_statistic(local_energies: Sequence[float]) -> float:
    """Return ``W = P99.9(abs(E_L - median(E_L)))`` (the instrument's estimator).

    ``contract_inputs.json``'s ``measurement.fresh_evaluation.estimator``:
    the order statistic at index ``ceil(0.999*N)`` (1-based) of the sorted
    absolute deviations from the stream's own median, no interpolation.

    Raises
    ------
    ValueError
        If *local_energies* is empty.
    """

    return _nearest_rank_p999(_deviations_about_own_median(local_energies))


def pooled_w_statistic(streams: Sequence[Sequence[float]]) -> float:
    """Return the pooled ``W`` across several streams (D1, review r2 delta).

    ``contract_inputs.json``'s ``measurement.fresh_evaluation.estimator``:
    "median: computed independently per stream". Each stream's absolute
    deviations are computed about ITS OWN median, then pooled across
    streams, then the nearest-rank P99.9 order statistic is taken over the
    pooled deviations -- never re-centered after pooling. A location shift
    applied to one stream alone (its deviations unchanged) must not change
    the result.

    Raises
    ------
    ValueError
        If any stream in *streams* is empty.
    """

    pooled: list[float] = []
    for stream in streams:
        pooled.extend(_deviations_about_own_median(stream))
    return _nearest_rank_p999(pooled)


def is_interior_minimum(c1: int) -> bool:
    """Return whether *c1* sits strictly inside the pilot window, not at its edge.

    Operator ruling (operator-rulings-pilot-20261110-1900, item 3): a basin
    center at the window edge (``c1`` of 3 or 19) never had a full
    neighboring triple on both sides, so it must never be reported as an
    interior minimum.

    Raises
    ------
    ValueError
        If *c1* is outside ``3..19``.
    """

    if not 3 <= c1 <= 19:
        raise ValueError(f"c1 must be in 3..19, got {c1}")
    return c1 not in (3, 19)


def select_pilot_basin_ordinal(stream0_w_by_ordinal: Mapping[int, float]) -> tuple[int, dict[int, float]]:
    """Return ``(c1, triple_means)`` -- the SEL step (pilot-plan-r3-20261110 (a)).

    *c1* is the center (ordinal 3..19) of the stream-0 triple with the
    smallest mean W; the EARLIEST (lowest-ordinal) tie wins, never the
    latest.

    Raises
    ------
    ValueError
        If *stream0_w_by_ordinal* is missing any ordinal a candidate triple needs.
    """

    centers = range(3, 20)
    missing = sorted({o for c in centers for o in (c - 1, c, c + 1) if o not in stream0_w_by_ordinal})
    if missing:
        raise ValueError(f"stream-0 W is missing for ordinals {missing!r}")
    triple_means = {
        c: (stream0_w_by_ordinal[c - 1] + stream0_w_by_ordinal[c] + stream0_w_by_ordinal[c + 1]) / 3.0
        for c in centers
    }
    c1 = min(triple_means, key=lambda c: (triple_means[c], c))
    return c1, triple_means


def select_final_basin_ordinal(c1: int, pooled01_w_by_ordinal: Mapping[int, float]) -> int:
    """Return ``c*``, the final basin ordinal (pilot-plan-r3 (a)).

    Candidates are the three centers ``{c1-1, c1, c1+1}`` clipped to
    ``3..19``; *c1* is always among them. Pools streams 0 and 1; the
    earliest tie wins.

    Raises
    ------
    ValueError
        If *pooled01_w_by_ordinal* is missing any ordinal a candidate triple needs.
    """

    candidates = sorted({c for c in (c1 - 1, c1, c1 + 1) if 3 <= c <= 19} | {c1})
    missing = sorted({o for c in candidates for o in (c - 1, c, c + 1) if o not in pooled01_w_by_ordinal})
    if missing:
        raise ValueError(f"pooled stream-0/1 W is missing for ordinals {missing!r}")
    means = {
        c: (pooled01_w_by_ordinal[c - 1] + pooled01_w_by_ordinal[c] + pooled01_w_by_ordinal[c + 1]) / 3.0
        for c in candidates
    }
    return min(means, key=lambda c: (means[c], c))


def assert_hydrogen_contract(config: dict[str, Any], *, evaluation: bool = False) -> None:
    """Refuse configs that drift from one-electron, charge-one hydrogen."""

    system = config.get("system", {})
    if system.get("n_particles") != 1 or system.get("spin") != {"n_up": 1, "n_down": 0}:
        raise ValueError("H-v2 requires exactly one spin-up electron")
    nuclei = system.get("nuclei", {})
    if nuclei.get("positions") != [[0.0, 0.0, 0.0]] or nuclei.get("charges") != [1.0]:
        raise ValueError("H-v2 requires one charge-one nucleus at the origin")
    terms = config.get("hamiltonian_terms", {})
    if set(terms) != {"kinetic", "electron_nucleus"}:
        raise ValueError("H-v2 Hamiltonian must contain only kinetic and electron-nucleus terms")
    if terms["kinetic"].get("_target_") != "tpen.physics.kinetic.KineticEnergy":
        raise ValueError("H-v2 requires the -1/2 kinetic-energy operator")
    if terms["electron_nucleus"].get("_target_") != "tpen.physics.potential.ElectronNucleusPotential":
        raise ValueError("H-v2 requires the raw electron-nucleus potential")
    if terms["electron_nucleus"].get("eps") != 0.0:
        raise ValueError("H-v2 Coulomb potential must be raw -1/r")
    sampler = config.get("sampler", {})
    if sampler.get("_target_") != "tpen.sampling.metropolis.MetropolisSampler":
        raise ValueError("H-v2 requires the Metropolis sampler")
    factors = config.get("model", {}).get("factors", [])
    targets = [factor.get("_target_") for factor in factors]
    if targets != ["tpen.nn.ElectronNucleusCusp"]:
        raise ValueError("H-v2 model must have only the nuclear cusp factor")
    if "law" not in factors[0]:
        raise ValueError("H-v2 nuclear cusp factor must declare a law")
    law = factors[0]["law"]
    if not isinstance(law, dict):
        raise ValueError("H-v2 nuclear cusp law must be a mapping")
    if law.get("_target_") != CONTRACT_NUCLEAR_LAW["_target_"]:
        raise ValueError("H-v2 nuclear cusp law must be the contract curvature law")
    if law.get("curvature_coefficient") != CONTRACT_NUCLEAR_LAW["curvature_coefficient"]:
        raise ValueError("H-v2 nuclear cusp law coefficient must be 0.01")
    if law.get("curvature_range") != CONTRACT_NUCLEAR_LAW["curvature_range"]:
        raise ValueError("H-v2 nuclear cusp law range must be 1.0")
    if law.get("trainable") is not CONTRACT_NUCLEAR_LAW["trainable"]:
        raise ValueError("H-v2 nuclear cusp law must be trainable")
    if law != CONTRACT_NUCLEAR_LAW:
        raise ValueError("H-v2 nuclear cusp law carries unexpected keys")
    if not evaluation and config.get("sampler", {}).get("seed") == config.get("runtime", {}).get("seed"):
        raise ValueError("training and sampler seeds must be distinct")
    if evaluation:
        if config.get("load", {}).get("mode") != "model_only":
            raise ValueError("fresh evaluation must request model-only restore")
        if "optimizer" in config or "trainer" in config:
            raise ValueError("fresh evaluation must not carry training state")
    else:
        _assert_a1_callbacks(config)
    assert_amplitude_risk_thresholds(config)


def _assert_a1_callbacks(config: Mapping[str, Any]) -> None:
    """Validate the complete appended A1 callback set and its ordering."""

    callbacks = [entry for entry in config.get("callbacks", ()) if isinstance(entry, Mapping)]
    present = [entry.get("_target_") for entry in callbacks]
    appended = set(present) - set(_A1_CANDIDATES_PRESENT_IN_BASE_CONFIG)
    if not appended:
        return
    missing = set(A1_TRAINING_MODULE_CANDIDATES) - set(present)
    if missing:
        raise ValueError(f"H-v2 A1 callback set is missing {sorted(missing)!r}")
    checkpoint = present.index("tpen.callback.Checkpoint")
    health = {
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.SamplerHealth",
        "tpen.callback.FactorScalars",
        "tpen.callback.RawTrainingBatchRecorder",
    }
    late = [target for index, target in enumerate(present) if target in health and index > checkpoint]
    if late:
        raise ValueError(f"H-v2 health callbacks must precede Checkpoint: {late!r}")
    expected_groups = {name: list(paths) for name, paths in module_norm_groups(config).items()}
    for entry in callbacks:
        if entry.get("_target_") in {"tpen.callback.ModuleNorms", "tpen.callback.OptimizerState"}:
            if entry.get("every_n_steps") != HEALTH_CALLBACK_CADENCE or entry.get("groups") != expected_groups:
                raise ValueError("H-v2 grouped callback cadence or groups drifted")
        if entry.get("_target_") == "tpen.callback.RawTrainingBatchRecorder":
            if entry.get("every_n_steps") != TRAINING_BATCH_RECORDER_CADENCE:
                raise ValueError("H-v2 training recorder cadence must be 50 iterations")
            if entry.get("partition_size_steps") != TRAINING_HISTORY_PARTITION_SIZE:
                raise ValueError("H-v2 training history partitions must be 1000 iterations")


def assert_amplitude_risk_thresholds(config: dict[str, Any]) -> None:
    """Refuse a configured ``AmplitudeRiskFraction`` whose thresholds drift.

    A no-op when ``config`` carries no such callback entry: this module adds
    no gate of its own, it only enforces agreement, once such a callback is
    present, with the frozen study list `AMPLITUDE_RISK_THRESHOLDS` decided by
    note ``amplitude-risk-call-20261106``.
    """

    callbacks = config.get("callbacks") or []
    for entry in callbacks:
        if not isinstance(entry, dict):
            continue
        if entry.get("_target_") != "tpen.callback.AmplitudeRiskFraction":
            continue
        thresholds = entry.get("thresholds")
        if list(thresholds) != list(AMPLITUDE_RISK_THRESHOLDS):
            raise ValueError(
                "H-v2 amplitude-risk callback thresholds must equal the frozen "
                f"study list {AMPLITUDE_RISK_THRESHOLDS}, got {thresholds!r}"
            )


__all__ = [
    "A1_TRAINING_MODULE_CANDIDATES",
    "AMPLITUDE_RISK_LABELS",
    "AMPLITUDE_RISK_PRIMARY_THRESHOLD",
    "AMPLITUDE_RISK_THRESHOLDS",
    "CONTRACT_INPUTS_PATH",
    "CONTRACT_NUCLEAR_LAW",
    "ENDQ_A1_QUANTILES",
    "EVALUATION_SEEDS",
    "EVAL_FORBIDDEN_KEYS",
    "EVAL_SCALE_OVERLAY_KEYS",
    "EVAL_SCALE_OVERLAY_PATH",
    "HEALTH_CALLBACK_CADENCE",
    "MODULE_NORM_GROUPS",
    "PILOT_A1_CADENCE",
    "PILOT_PAYLOAD_OVERLAY_KEYS",
    "PILOT_PAYLOADS",
    "PILOT_STAGE1_ORDINALS",
    "PILOT_STAGE1_STREAM",
    "READOUT_EPS",
    "READOUT_TRAJECTORY_GENERATOR_TARGET",
    "ROOT",
    "SAMPLER_SEED",
    "SCIENCE_SCALE_OVERLAY_KEYS",
    "SCIENCE_SCALE_OVERLAY_PATH",
    "TRAINING_SEED",
    "TRAINING_BATCH_RECORDER_CADENCE",
    "TRAINING_HISTORY_PARTITION_SIZE",
    "UNSPECIFIED_MODULES",
    "append_candidate_module",
    "append_measurement_callbacks",
    "assert_amplitude_risk_thresholds",
    "assert_hydrogen_contract",
    "candidate_modules_to_append",
    "enable_readout_trajectory_generator",
    "evaluation_config",
    "gpu_route_g_eval_seed",
    "is_interior_minimum",
    "load_yaml",
    "pilot_eval_seed",
    "pilot_stage1_grid",
    "pilot_stage2_grid",
    "pilot_stage2_window",
    "module_norm_groups",
    "pilot_training_config",
    "pooled_w_statistic",
    "reconcile_enabled_modules",
    "remove_candidate_module",
    "science_eval_seed",
    "science_evaluated_checkpoint_updates",
    "science_evaluation_config",
    "science_training_config",
    "select_final_basin_ordinal",
    "select_pilot_basin_ordinal",
    "training_config",
    "unspecified_modules",
    "w_statistic",
]
