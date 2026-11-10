"""Frozen H-v2 pipeline configuration helpers and identity checks."""

from __future__ import annotations

import copy
import json
import math
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

#: TO 63690220 amendment r3: every "A1 training-side logging module"
#: candidate the pilot tries to enable at A1 cadence. Admission is decided
#: per candidate, never as a block, by probing the runtime recovery profile
#: (:func:`experiments.toolkit.attached_activation.check_profile`) -- never
#: assumed from the branch. ``DataIntegrity``/``GradientStats`` are already
#: present leaves in ``train.yaml`` (cadence set by the payload overlay
#: above); the rest are appended fresh by :func:`append_measurement_callbacks`.
#: Amendment r4: ``ModuleNorms`` is deliberately NOT a candidate here. Its
#: required ``groups`` mapping names model-structure paths that are not yet
#: frozen (placeholders until 6c29fe1d/69c5a5a6), so there is no value this
#: module could honestly supply; appending it with a placeholder would pass
#: profile admission (a syntax-only check) and then fail at instantiation.
#: It is recorded separately, always, under :data:`UNSPECIFIED_MODULES`.
A1_TRAINING_MODULE_CANDIDATES: tuple[str, ...] = (
    "tpen.callback.DataIntegrity",
    "tpen.callback.GradientStats",
    "tpen.callback.InitialCheckpoint",
    "tpen.callback.ParameterActivity",
    "tpen.callback.ElectronNucleusDistanceQuantiles",
)

#: Amendment r4 rule 2: module candidates never probed or appended, recorded
#: unconditionally on the TRAIN receipt's ``skipped_unspecified`` list with
#: their reason (the TO item whose merge would specify them).
UNSPECIFIED_MODULES: dict[str, str] = {"tpen.callback.ModuleNorms": "6c29fe1d"}

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


def candidate_modules_to_append() -> tuple[str, ...]:
    """Return the A1 training-side module candidates not already in ``train.yaml``."""

    return tuple(
        target
        for target in A1_TRAINING_MODULE_CANDIDATES
        if target not in _A1_CANDIDATES_PRESENT_IN_BASE_CONFIG
    )


def append_candidate_module(config: dict[str, Any], target: str) -> None:
    """Append one candidate module's callback entry onto *config* in place."""

    config["callbacks"].append(_build_candidate_callback(target))


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


#: Design target C2's frozen seed formula (contract_inputs.json
#: ``measurement.seed_roster.evaluation_stream_seed_formula``), restated as
#: code rather than a second literal copy.
def science_eval_seed(lineage_index: int, checkpoint_ordinal: int, stream_index: int) -> int:
    """Return one science evaluation generation's frozen seed (design target C2)."""

    return 1_000_000 + 10_000 * lineage_index + 10 * checkpoint_ordinal + stream_index


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

    values = sorted(local_energies)
    n = len(values)
    if n == 0:
        raise ValueError("W requires at least one local energy")
    median = values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    deviations = sorted(abs(value - median) for value in local_energies)
    index = min(math.ceil(0.999 * n), n) - 1
    return deviations[index]


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


__all__ = [
    "A1_TRAINING_MODULE_CANDIDATES",
    "CONTRACT_INPUTS_PATH",
    "CONTRACT_NUCLEAR_LAW",
    "ENDQ_A1_QUANTILES",
    "EVALUATION_SEEDS",
    "EVAL_FORBIDDEN_KEYS",
    "EVAL_SCALE_OVERLAY_KEYS",
    "EVAL_SCALE_OVERLAY_PATH",
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
    "UNSPECIFIED_MODULES",
    "append_candidate_module",
    "append_measurement_callbacks",
    "assert_hydrogen_contract",
    "candidate_modules_to_append",
    "enable_readout_trajectory_generator",
    "evaluation_config",
    "gpu_route_g_eval_seed",
    "load_yaml",
    "pilot_eval_seed",
    "pilot_stage1_grid",
    "pilot_stage2_grid",
    "pilot_stage2_window",
    "pilot_training_config",
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
