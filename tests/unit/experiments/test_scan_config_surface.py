"""The tpen-pair-scan-v1 slot surface holds the grid's contracts.

The scan drives 288 runs from two base configs through a handful of dotted
override paths. Everything here guards a property that, if it broke, would
produce a green grid whose numbers answer a different question than the one
asked. None of it is style.

The properties, and what each one is defending against:

channels fan-out
    tpen-pair-v1 spells the feature width as four independent literal ``4``s
    (``train.yaml:56,66,75,91``), one per consumer. There is no single override
    that moves all four and no error if only three move. Here one
    ``run_parameters.channels`` reaches every consumer through
    ``model_params.channels``, and a config in which the four could disagree is
    the failure mode these tests exist to make unrepresentable.

embedding activation held at SiLU
    ``activation_slot`` owns the equivariant-op activation only (D2). The
    embedding MLP is a plain feedforward net -- a separate concept -- and is a
    control. Both the config text and the constructed modules are checked,
    because a leak would move two unrelated things per axis step and make the
    activation axis uninterpretable.

three independent seeds
    ``sampler.seed: ${runtime.seed}`` (``train.yaml:28,108``) fuses model
    initialization and the sampler chain. Paired seed rows need validation to
    reproduce a train row's MODEL while drawing a fresh sampler stream, which one
    integer cannot express.

omega / confinement coefficient
    ``coefficient = omega / 2`` with no arithmetic resolver in
    ``tpen/config.py``, so one number is pinned in four places across two files.
    Asserted, not promised by a comment.

train / eval model identity
    Validation restores under ``mode: model_only, strict: true``, which rejects
    any wiring difference.

callbacks at config root
    ``_instantiate_runner`` raises on a runner that declares ``callbacks`` or
    ``loggers`` (``tpen/run.py:343-348``). The mutant that moves them is checked
    against the real guard, not only structurally.

metric provenance
    ``MCMCGenerator`` draws from |psi|^2, so its ``local_energy_mean`` is the
    variational estimator. ``StratifiedGeometryGenerator`` does ``del model``
    (``generators/hooke.py:232``) and draws from a fixed prior, so its mean is
    E_q[E_L] for arbitrary q and is unbounded below; only its VARIANCE is a valid
    objective there. The tests pin which task supplies which key.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations, LocalEnergyValues
from tpen.evaluation.protocols import EvaluationContext
from tpen.run import run_from_config

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY = REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml"
CONFIG_DIR = REPO_ROOT / "experiments" / "hooke" / "tpen-pair-scan-v1" / "configs"
STAGES = ("train", "eval")

# The axis levels. Basis levels are NOT redeclared here or in the scan configs:
# they come from the one shared library, and this tuple only names them.
BASIS_LEVELS = ("no-basis", "hooke-axiswise-v1", "hooke-total-shell", "hooke-cartesian-box")
ACTIVATION_LEVELS = ("SiLU", "Tanh", "Sigmoid", "Gaussian")
CHANNELS_LEVELS = (8, 16, 32)
LR_LEVELS = (1.0e-3, 3.0e-3)

# Per-particle one-body width each basis level presents to the embedding, at
# spatial_dim 3 with spin included. Hard-coded rather than recomputed from the
# enumerators: a test that re-derives a width from the code under test cannot
# detect a change in that code. The cartesian-box entry is 9, not the 8
# multi-indices the box admits -- ``ElectronBasis.out_features`` adds the spin
# channel (``tpen/nn/basis.py:143-147``).
EXPECTED_IN_FEATURES = {
    "no-basis": None,
    "hooke-axiswise-v1": 7,
    "hooke-total-shell": 5,
    "hooke-cartesian-box": 9,
}

# The four consumers one ``channels`` override has to reach.
CHANNELS_CONSUMERS = (
    "model.embedding.out_channels",
    "model.layers[0].mixing.channels",
    "model.layers[0].path_aggregation.channels",
    "model.readout.channels",
)

# The two consumers ``activation_slot`` is allowed to reach, and the module each
# level must produce there.
ACTIVATION_CONSUMERS = (
    "model.layers[0].mixing.activation",
    "model.layers[0].path_aggregation.activation",
)
ACTIVATION_TARGETS = {
    "SiLU": "torch.nn.SiLU",
    "Tanh": "torch.nn.Tanh",
    "Sigmoid": "torch.nn.Sigmoid",
    "Gaussian": "tpen.nn.GaussianActivation",
}
ACTIVATION_TYPES = {
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "Gaussian": None,  # resolved lazily; tpen.nn import is deferred below
}

# The whole public override surface. An axis added or a seed dropped changes this
# tuple, which is the point: the planner's vocabulary is a contract.
RUN_PARAMETER_KEYS = frozenset(
    {
        "basis_slot",
        "activation_slot",
        "lr",
        "channels",
        "training_model_seed",
        "training_sampler_seed",
        "validation_sampler_seed",
        "accelerator_synchronize",
    }
)

SEED_KEYS = ("training_model_seed", "training_sampler_seed", "validation_sampler_seed")

TIMING_CALLBACKS = {
    "train": (
        "tpen.callback.RunTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.TrainPhaseTiming",
    ),
    "eval": (
        "tpen.callback.EvaluationTiming",
        "tpen.callback.EvaluationComponentTiming",
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _raw(stage: str) -> DictConfig:
    """Return one stage config exactly as it is written on disk.

    Unmerged and unresolved: the tests that read config *text* -- literals versus
    interpolations -- must not see a resolver's answer.
    """

    return OmegaConf.load(CONFIG_DIR / f"{stage}.yaml")


def _config(stage: str, **overrides: Any) -> DictConfig:
    """Return one stage config merged with the basis library and overridden.

    Reproduces the real consumption path: the study config is merged with
    ``experiments/hooke/choices/basis_levels.yaml`` (``tpen.run.load_config``
    reads exactly one YAML file, with no include mechanism) and then given
    dotlist overrides exactly as ``run.py`` would.
    """

    cfg = OmegaConf.merge(_raw(stage), OmegaConf.load(LIBRARY))
    if overrides:
        # Booleans go through as YAML scalars: `f"{True}"` is "True", which
        # OmegaConf would be free to read as a string, and a string-typed knob
        # would then pass an equality check against the wrong type.
        dotlist = [
            f"run_parameters.{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in overrides.items()
        ]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def _select(cfg: DictConfig, path: str) -> Any:
    """Resolve one dotted path, tolerating the ``[0]`` list index spelling."""

    return OmegaConf.select(cfg, path.replace("[0]", ".0"))


def _raw_select(stage: str, path: str) -> Any:
    """Return the UNRESOLVED node at one dotted path of a stage config."""

    node = _raw(stage)
    for part in path.replace("[0]", ".0").split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    return node


def _raw_text(stage: str, path: str) -> str:
    """Return the raw interpolation string at one dotted path, or ``''``."""

    parent_path, _, leaf = path.rpartition(".")
    parent = _raw_select(stage, parent_path)
    value = OmegaConf.to_container(parent, resolve=False)[leaf]
    return value if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# The override surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_both_stages_expose_exactly_the_same_override_vocabulary(stage: str) -> None:
    """One vocabulary, both stages -- no stage-specific override paths.

    ``load_config`` applies overrides with
    ``OmegaConf.merge(cfg, OmegaConf.from_dotlist(...))`` (``tpen/run.py:88-90``),
    which SILENTLY CREATES an unknown key rather than rejecting it. A path
    present in one stage and absent in the other therefore no-ops in silence on
    the stage that lacks it, with no error anywhere. Equal key sets is the only
    thing that makes a stray override path impossible.
    """

    assert frozenset(_raw(stage).run_parameters.keys()) == RUN_PARAMETER_KEYS


@pytest.mark.parametrize("stage", STAGES)
def test_basis_levels_are_selected_from_the_shared_library_never_redeclared(stage: str) -> None:
    """The scan selects from one basis table; it does not own a copy.

    A redeclared level would make "the scan ran the level we tested" a claim
    about two files that were once identical. The scan config must therefore have
    no ``choices.basis`` of its own, and the merged config must expose exactly
    the library's four levels.
    """

    assert OmegaConf.select(_raw(stage), "choices.basis") is None
    assert tuple(_config(stage).choices.basis.keys()) == BASIS_LEVELS


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("slot", BASIS_LEVELS)
def test_embedding_input_width_comes_from_the_selected_basis_slot(stage: str, slot: str) -> None:
    """``in_features`` tracks the basis, and is a slot key rather than a resolver.

    ``in_features: ${tpen.basis_feature_dim:${model.basis}}`` raises ``TypeError``
    on ``no-basis``: the resolver instantiates its argument and reads
    ``out_features``, and ``instantiate(None)`` has neither
    (``tpen/config.py:66-70``). Resolving to ``null`` inside the slot is what lets
    the embedding fall back to its derived width 3 + 1 = 4.
    """

    cfg = _config(stage, basis_slot=slot)

    assert _select(cfg, "model.embedding.in_features") == EXPECTED_IN_FEATURES[slot]
    assert "basis_feature_dim" not in _raw_text(stage, "model.embedding.in_features")
    assert "run_parameters.basis_slot" in _raw_text(stage, "model.embedding.in_features")


# ---------------------------------------------------------------------------
# channels: one override, four consumers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("channels", CHANNELS_LEVELS)
def test_one_channels_override_reaches_every_consumer(stage: str, channels: int) -> None:
    """A single ``run_parameters.channels`` sets all four widths at once."""

    cfg = _config(stage, channels=channels)
    resolved = {path: _select(cfg, path) for path in CHANNELS_CONSUMERS}

    assert resolved == {path: channels for path in CHANNELS_CONSUMERS}


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("path", CHANNELS_CONSUMERS)
def test_no_channels_consumer_carries_a_literal_width(stage: str, path: str) -> None:
    """Every consumer interpolates; none of the four literal ``4``s survives.

    Read from the unresolved text, so a site pinned back to a literal fails here
    even when it happens to equal the default. The interpolation chain is checked
    end to end: consumer -> ``model_params.channels`` -> ``run_parameters.channels``.
    """

    text = _raw_text(stage, path)

    assert text == "${model_params.channels}", f"{path} does not interpolate the shared width"
    assert _raw_text(stage, "model_params.channels") == "${run_parameters.channels}"


@pytest.mark.parametrize("stage", STAGES)
def test_channels_consumers_cannot_be_made_to_disagree(stage: str) -> None:
    """No override drives one consumer without driving the other three.

    The four widths have to agree for the model to build at all -- the embedding
    output feeds mixing, and the readout asserts its own channel count against
    the order-2 block (``readout/pfaffian.py:148-152``). Sharing one source is
    what turns "they agree" from a thing to remember into a thing to read.
    """

    sources = {_raw_text(stage, path) for path in CHANNELS_CONSUMERS}

    assert len(sources) == 1


# ---------------------------------------------------------------------------
# The activation axis, and the embedding it must not touch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("slot", ACTIVATION_LEVELS)
def test_activation_slot_drives_both_equivariant_stages(stage: str, slot: str) -> None:
    """One slot sets the mixing and aggregation activation, and only those."""

    cfg = _config(stage, activation_slot=slot)

    for path in ACTIVATION_CONSUMERS:
        assert _select(cfg, f"{path}._target_") == ACTIVATION_TARGETS[slot]


@pytest.mark.parametrize("stage", STAGES)
def test_the_activation_library_defines_exactly_the_four_levels(stage: str) -> None:
    """Four levels; adding, dropping, or renaming one is a change to the axis."""

    assert tuple(_raw(stage).choices.activation.keys()) == ACTIVATION_LEVELS


@pytest.mark.parametrize("stage", STAGES)
def test_gaussian_level_is_the_elementwise_activation_not_the_decay_gate(stage: str) -> None:
    """The Gaussian level is ``exp(-x^2 / 2 sigma^2)`` with sigma fixed at 1.

    ``GaussianDecayGate`` is ``exp(-x / 2 sigma^2)`` on non-negative squared
    radii and DIVERGES on the signed feature blocks an equivariant stage carries,
    so substituting it would not fail loudly -- it would produce a level whose
    features blow up on half its inputs.
    """

    cfg = _config(stage, activation_slot="Gaussian")

    for path in ACTIVATION_CONSUMERS:
        assert _select(cfg, f"{path}._target_") == "tpen.nn.GaussianActivation"
        assert _select(cfg, f"{path}.sigma") == 1.0


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("slot", ACTIVATION_LEVELS)
def test_embedding_activation_is_silu_at_every_activation_level(stage: str, slot: str) -> None:
    """The embedding MLP is a held control, pinned to SiLU on all four levels.

    D2 gives the TPEN activation contract to ``EquivariantMixing`` and
    ``PathAggregation``; the embedding MLP merely happens to accept an
    ``activation`` kwarg. Sweeping both would move two unrelated things per axis
    step. Held fixed, the axis isolates the novel TPEN surface.
    """

    cfg = _config(stage, activation_slot=slot)

    assert _select(cfg, "model.embedding.activation._target_") == "torch.nn.SiLU"


@pytest.mark.parametrize("stage", STAGES)
def test_embedding_activation_is_a_written_literal_not_a_slot_interpolation(stage: str) -> None:
    """The control is spelled out, so a leak is visible in the config text.

    Two failure modes, one assertion each. Interpolating the slot here would make
    the embedding a second, undeclared consumer of the activation axis. Deleting
    the line would leave the same value by ``tpen/nn/mlp.py:61``'s default and be
    entirely unobservable -- which is exactly what the layer-3 basis library
    found for ``include_gaussian_factor``, where the explicitness was the only
    guard that existed.
    """

    embedding = OmegaConf.to_container(_raw_select(stage, "model.embedding"), resolve=False)

    assert embedding["activation"] == {"_target_": "torch.nn.SiLU"}
    assert "activation_slot" not in OmegaConf.to_yaml(_raw_select(stage, "model.embedding"))


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("slot", ACTIVATION_LEVELS)
def test_constructed_embedding_mlps_use_silu_at_every_activation_level(
    stage: str, slot: str
) -> None:
    """The built modules agree with the config: SiLU in the embedding, slot in the ops.

    The config-text tests above cannot see a wiring mistake that resolves to the
    right ``_target_`` but reaches the wrong module. This one builds the model and
    reads the activations off the constructed MLPs and equivariant stages.
    """

    from tpen.nn import GaussianActivation

    expected_op = GaussianActivation if slot == "Gaussian" else ACTIVATION_TYPES[slot]
    model = instantiate(_config(stage, activation_slot=slot).model)

    embedding_activations = [
        module
        for mlp in model.embedding.order_mlps.values()
        for module in mlp.layers
        if not isinstance(module, nn.Linear)
    ]
    assert embedding_activations, "the embedding MLPs have no activation to check"
    assert all(isinstance(module, nn.SiLU) for module in embedding_activations)

    layer = model.stack.layers[0]
    assert isinstance(layer.mixing.activation, expected_op)
    assert isinstance(layer.path_aggregation.activation, expected_op)


# ---------------------------------------------------------------------------
# Three independent seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_both_stages_declare_the_full_seed_triple(stage: str) -> None:
    """All three seeds exist in both stages even where one is unread.

    A planner emits one seed triple per grid row. Because an override of an
    undeclared key is silently created rather than rejected, a stage missing one
    of the three would accept the override and ignore it.
    """

    assert all(key in _raw(stage).run_parameters for key in SEED_KEYS)


def test_train_seeds_reach_their_own_consumers() -> None:
    """Model initialization and the training chain read different integers.

    tpen-pair-v1 has ``sampler.seed: ${runtime.seed}`` (``train.yaml:28,108``),
    which fuses them. Distinct values here prove the split; equal values would
    pass under the fused wiring too, which is why none of the three is repeated.
    """

    cfg = _config("train", training_model_seed=11, training_sampler_seed=22, validation_sampler_seed=33)

    assert cfg.runtime.seed == 11
    assert cfg.sampler.seed == 22


def test_validation_sampler_seed_is_the_only_seed_the_eval_stage_samples_with() -> None:
    """Validation replays the train model seed and swaps only the sampler stream.

    That pairing is the whole point of the paired seed rows: same model, fresh
    chain. It requires the evaluation sampler to read a seed that the train stage
    does not use for anything else.
    """

    cfg = _config("eval", training_model_seed=11, training_sampler_seed=22, validation_sampler_seed=33)

    assert cfg.runtime.seed == 11
    assert cfg.evaluation_sampler.seed == 33
    assert cfg.evaluator.tasks[0].generator.seed == 33


@pytest.mark.parametrize("moved", SEED_KEYS)
def test_moving_one_seed_moves_nothing_else(moved: str) -> None:
    """Each seed is separately addressable: one override, one consumer moves.

    Collapsing any two seeds onto one source would leave a row's model and its
    sampler stream co-varying, and the seed replicates would then measure
    initialization noise and sampling noise as one quantity.
    """

    baseline = dict.fromkeys(SEED_KEYS, 0)
    perturbed = {**baseline, moved: 7}
    consumers = {
        "training_model_seed": ("train", "runtime.seed"),
        "training_sampler_seed": ("train", "sampler.seed"),
        "validation_sampler_seed": ("eval", "evaluation_sampler.seed"),
    }

    for key, (stage, path) in consumers.items():
        before = _select(_config(stage, **baseline), path)
        after = _select(_config(stage, **perturbed), path)
        if key == moved:
            assert (before, after) == (0, 7), f"{path} did not follow {moved}"
        else:
            assert after == before, f"{path} moved when only {moved} was overridden"


def test_fixed_geometry_evaluation_points_do_not_move_with_a_row_seed() -> None:
    """Every grid row is scored on the SAME fixed-prior points.

    The stratified, cusp, tail, orbital and permutation-orbit tasks compare a
    wavefunction on a point set. Seeding them from the row's sampler seed would
    evaluate a row's three seed replicates on three different geometries, mixing
    sampling noise into a fixed-geometry diagnostic.
    """

    tasks = ("stratified_geometry", "cusp", "tail", "hooke_orbital")
    moved = _config("eval", validation_sampler_seed=99)
    still = _config("eval", validation_sampler_seed=0)

    for task in tasks:
        path = f"evaluation_tasks.{task}.generator.seed"
        assert _select(moved, path) == _select(still, path) == moved.evaluation.geometry_seed


# ---------------------------------------------------------------------------
# omega and the confinement coefficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_confinement_coefficient_is_half_the_system_omega(stage: str) -> None:
    """``coefficient == omega / 2``, asserted rather than commented.

    There is no arithmetic resolver in ``tpen/config.py``, so the two values are
    pinned independently -- twice per file, across two files. omega is not an
    axis of this scan; if it ever becomes one, this coupling has to be solved
    before the axis can exist.
    """

    cfg = _config(stage)
    envelopes = cfg.model.envelope.envelopes
    confinement = [item for item in envelopes if item._target_.endswith("GaussianConfinement")]

    assert len(confinement) == 1
    assert confinement[0].coefficient == pytest.approx(cfg.system.omega / 2)


def test_both_stages_pin_the_same_omega() -> None:
    """A per-stage omega would restore a checkpoint into a different Hamiltonian."""

    assert _config("train").system.omega == _config("eval").system.omega


# ---------------------------------------------------------------------------
# train / eval model identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("basis_slot", "activation_slot", "channels"),
    tuple(itertools.product(BASIS_LEVELS, ACTIVATION_LEVELS, CHANNELS_LEVELS)),
)
def test_train_and_eval_model_blocks_resolve_identically(
    basis_slot: str, activation_slot: str, channels: int
) -> None:
    """Every point of the slot grid resolves to one model wiring in both stages.

    Validation restores with ``mode: model_only, strict: true``, so any
    difference -- a width, an activation, an envelope coefficient -- turns into a
    restore failure at validation time, after the training cost has been paid.
    Checked over the whole 4 x 4 x 3 grid rather than a sample, because it is
    pure config resolution and a per-slot divergence is exactly the kind a sample
    misses.
    """

    overrides = {
        "basis_slot": basis_slot,
        "activation_slot": activation_slot,
        "channels": channels,
    }
    train = OmegaConf.to_container(_config("train", **overrides).model, resolve=True)
    evaluation = OmegaConf.to_container(_config("eval", **overrides).model, resolve=True)

    assert train == evaluation


# ---------------------------------------------------------------------------
# Callbacks, loggers and timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("forbidden", ("callbacks", "loggers"))
def test_runner_never_owns_callbacks_or_loggers(stage: str, forbidden: str) -> None:
    """Callbacks and loggers are config-root and ``RunContext``-owned."""

    assert forbidden not in _raw(stage).runner
    assert forbidden in _raw(stage)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("forbidden", ("callbacks", "loggers"))
def test_moving_callbacks_into_the_runner_is_rejected_by_the_real_guard(
    stage: str, forbidden: str, tmp_path: Path
) -> None:
    """The structural check above is backed by the guard that actually runs.

    ``_instantiate_runner`` raises ``ValueError`` for a runner declaring either
    key (``tpen/run.py:343-348``), and ``run_from_config`` turns that into exit
    code 1. Exercising it on THIS config, rather than on a synthetic one, is what
    proves the rejection applies to the scan's runner block.
    """

    cfg = _config(stage)
    cfg.runner[forbidden] = cfg.pop(forbidden)
    cfg.run.root = str(tmp_path)
    # CPU so the guard can be reached without an accelerator; the guard fires
    # before any component is constructed.
    cfg.runtime.device = "cpu"
    if stage == "eval":
        cfg.load.path = str(tmp_path / "absent_checkpoints")

    assert run_from_config(cfg, config_path=str(CONFIG_DIR / f"{stage}.yaml"), command="pytest") == 1


@pytest.mark.parametrize("stage", STAGES)
def test_timing_callbacks_are_registered_at_config_root(stage: str) -> None:
    """The timing callbacks exist, at the top level, in the right stage.

    tpen-pair-v1 registers none of them, which is why a real smoke run's
    ``metrics.csv`` carried no timing keys at all: the callbacks were always
    available (``tpen/callback/timing/``), the configs simply never named them.
    Without them a 288-run cost table is blank in every per-phase column.
    """

    targets = [entry._target_ for entry in _raw(stage).callbacks]

    for expected in TIMING_CALLBACKS[stage]:
        assert expected in targets, f"{stage}.yaml does not register {expected}"


@pytest.mark.parametrize("stage", STAGES)
def test_every_configured_callback_constructs(stage: str) -> None:
    """The callback list is instantiable, kwargs included.

    A timing callback given a keyword it does not accept raises ``TypeError`` at
    construction -- i.e. after the queue wait, on the compute node. Cheap to
    catch here.
    """

    cfg = _config(stage)
    cfg.run.dir = "/tmp/scan-config-surface"

    assert [instantiate(entry) for entry in cfg.callbacks]


@pytest.mark.parametrize("stage", STAGES)
def test_accelerator_synchronize_defaults_false_and_flips_from_one_path(stage: str) -> None:
    """Off for the grid, on for the probe, from a single override.

    ``TrainStepTiming`` synchronizes the device at both scope boundaries
    (``train_step_timing.py:61,72``). Doing that every step perturbs the quantity
    being measured and serializes work that would otherwise overlap, so the
    production grid runs with it off; the timing probe, where attribution matters
    more than throughput, turns it on without editing the config.
    """

    default = _config(stage)
    flipped = _config(stage, accelerator_synchronize=True)
    timing_targets = set(TIMING_CALLBACKS[stage])

    assert default.run_parameters.accelerator_synchronize is False
    for cfg, expected in ((default, False), (flipped, True)):
        registered = [entry for entry in cfg.callbacks if entry._target_ in timing_targets]
        assert len(registered) == len(timing_targets)
        for entry in registered:
            assert entry.accelerator_synchronize is expected, f"{entry._target_} ignored the knob"


def test_the_terminal_checkpoint_uses_an_explicit_terminal_only_schedule() -> None:
    """Validation can restore whatever ``max_steps`` the probe picks.

    The stream declares ``TerminalOnly`` explicitly, so no periodic cadence can
    suppress the terminal boundary.
    """

    checkpoint = [
        entry
        for entry in _raw("train").callbacks
        if entry._target_ == "tpen.callback.Checkpoint"
    ]

    assert len(checkpoint) == 1
    assert checkpoint[0].terminal is True
    assert checkpoint[0].schedule._target_ == "tpen.checkpoint.TerminalOnly"
    assert checkpoint[0].payload._target_ == "tpen.checkpoint.TrainResume"
    assert "every_n_steps" not in checkpoint[0], (
        "legacy every_n_steps must not replace the explicit terminal-only schedule"
    )
    assert "every_n_steps" not in _raw("train").checkpoint


def test_the_eval_stage_requires_an_injected_checkpoint_path() -> None:
    """`load.path` is mandatory-missing, so a forgotten injection fails loudly.

    A default would be worse than no value: validation would restore some other
    run's model and report plausible numbers for the wrong row.
    """

    assert OmegaConf.is_missing(_raw("eval").load, "path")
    assert _raw("eval").load.mode == "model_only"
    assert _raw("eval").load.strict is True


def test_the_timing_cadence_knob_is_passed_only_where_it_works() -> None:
    """`every_n_steps` throttles `TrainPhaseTiming` and is inert on `TrainStepTiming`.

    Measured on a 60-step GPU smoke: 6 phase rows, 60 step rows, from one
    ``every_n_steps: 10``. The asymmetry is real. `TrainPhaseTiming` pops the
    kwarg and builds its own report `Cadence` (``train_phase_timing.py:64-74``);
    `TrainStepTiming` forwards it to the base, which stores it as an attribute
    (``callback/base.py:104``) while typed delivery is gated by
    ``SubscriptionGroup(cadence=...)`` — and that callback builds its group
    without one, so every occurrence is delivered.

    So the kwarg is passed to the callback that honours it and withheld from the
    one that does not. Passing it to both would read as a throttle on the step
    stream and silently be nothing, which is worse than the untuned default: it
    would make a ~500-row-per-run `train/perf` stream look like a ~50-row one in
    any cost estimate built from the config.
    """

    registered = {entry._target_: entry for entry in _raw("train").callbacks}
    step_timing = registered["tpen.callback.TrainStepTiming"]
    phase_timing = registered["tpen.callback.TrainPhaseTiming"]

    assert "every_n_steps" not in step_timing, (
        "every_n_steps is inert on TrainStepTiming; passing it claims a throttle "
        "that does not exist"
    )
    assert step_timing.rolling_window >= 20, "the rolling mean is what carries the signal"
    assert phase_timing.every_n_steps == _config("train").timing.every_n_steps
    assert _config("train").timing.every_n_steps > 1


# ---------------------------------------------------------------------------
# Metric provenance: which task supplies which selection key
# ---------------------------------------------------------------------------


def _local_energy_metrics(
    summary_cfg: DictConfig, values: torch.Tensor, tmp_path: Path
) -> dict[str, Any]:
    """Return the metrics one configured summary emits for given local energies.

    A real `EvaluationContext` rather than ``None``: the suite runs under typeguard,
    which enforces the annotated parameter type at the call boundary.
    """

    bundle = EvaluationBundle(
        generated=GeneratedConfigurations(
            batch=ElectronBatch(
                positions=torch.zeros(values.numel(), 2, 3, dtype=torch.float64),
                spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64).repeat(values.numel(), 1),
            ),
            metadata={},
        ),
        local_energy=LocalEnergyValues(local_energy=values, finite_mask=torch.isfinite(values)),
    )
    context = EvaluationContext(
        namespace="eval",
        artifact_level="metrics_only",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=0,
        run_dir=tmp_path,
        task_output_dir=tmp_path,
        metadata={},
    )
    summary = instantiate(summary_cfg)
    return dict(summary.summarize(bundle=bundle, context=context, namespace="eval").metrics)


def test_the_primary_metric_comes_from_a_task_that_samples_the_wavefunction(tmp_path: Path) -> None:
    """``mcmc_energy`` supplies ``local_energy_mean``, and it is the variational one.

    ``MCMCGenerator`` draws from the trained sampler, i.e. from |psi|^2
    (``generators/mcmc.py:13-55``), so its mean local energy IS
    <psi|H|psi>/<psi|psi> and the variational principle bounds it below by the
    exact 2.0 Ha of the Hooke omega=0.5 singlet. That is what makes ``min`` on it
    the correct direction -- the property the fixed-prior task does not have.
    """

    cfg = _config("eval")
    task = cfg.evaluation_tasks.mcmc_energy

    assert cfg.evaluator.tasks[0].name == "mcmc_energy"
    assert task.generator._target_ == "tpen.evaluation.generators.MCMCGenerator"
    assert task.generator.sampler._target_ == "tpen.sampling.metropolis.MetropolisSampler"
    metrics = _local_energy_metrics(
        task.summaries[0], torch.tensor([2.0, 2.5, 1.5], dtype=torch.float64), tmp_path
    )
    assert "local_energy_mean" in metrics


def test_the_secondary_metric_is_a_variance_and_the_task_that_emits_it_is_kept(
    tmp_path: Path,
) -> None:
    """``stratified_geometry`` supplies ``local_energy_variance``, never its mean.

    ``StratifiedGeometryGenerator.generate`` begins ``del model``
    (``generators/hooke.py:232``) and draws from a fixed seeded prior q, so its
    ``local_energy_mean`` is E_q[E_L] for arbitrary q and is unbounded below:
    ``min`` on it rewards whichever wavefunction dips furthest below 2.0 on the
    sampled points. The decisive form is that for the exact eigenstate
    E_L(R) = 2.0 for every R, so E_q[E_L] = 2.0 for ANY q -- exactness pins the
    value at the optimum and gives no direction of approach. The VARIANCE is
    valid on the same prior, because Var_q[E_L] = 0 iff psi is an eigenstate for
    any q. Both keys come from the same summary, so this pins the key that exists
    rather than the one selection is allowed to use.
    """

    task = _config("eval").evaluation_tasks.stratified_geometry

    assert task.generator._target_ == "tpen.evaluation.generators.StratifiedGeometryGenerator"
    metrics = _local_energy_metrics(
        task.summaries[0], torch.tensor([2.0, 2.5, 1.5], dtype=torch.float64), tmp_path
    )
    assert "local_energy_variance" in metrics


def test_the_reference_energy_the_tiebreak_measures_is_the_exact_hooke_energy(
    tmp_path: Path,
) -> None:
    """``mcmc_energy`` also emits |mean - 2.0|, and 2.0 is not a magic number.

    The champion selector's tiebreak is the absolute deviation from the exact
    energy of the Hooke omega=0.5 singlet, and it is measured by
    ``ReferenceEnergySummary`` on the variational task rather than derived inside
    the selector. That matters in one direction the raw mean cannot express: a
    sampled energy that has fallen BELOW 2.0 is an estimator defect, and
    ``energy_abs_error`` penalizes it exactly as much as the same excursion above.

    There is no arithmetic resolver in ``tpen/config.py``, so the reference is a
    literal in the config; it is asserted against
    `tpen.physics.hooke.HookeSingletExact.exact_energy` so the literal cannot
    drift from the physics it claims to be.
    """

    from tpen.physics.hooke import HookeSingletExact

    task = _config("eval").evaluation_tasks.mcmc_energy
    reference_summaries = [
        summary
        for summary in task.summaries
        if summary._target_ == "tpen.evaluation.summaries.ReferenceEnergySummary"
    ]

    assert len(reference_summaries) == 1
    assert float(reference_summaries[0].reference_energy) == HookeSingletExact.exact_energy
    # The primary metric's summary keeps its position, so the two live side by
    # side rather than one replacing the other.
    assert task.summaries[0]._target_ == "tpen.evaluation.summaries.LocalEnergySummary"

    metrics = _local_energy_metrics(
        reference_summaries[0], torch.tensor([1.5, 2.0, 2.5], dtype=torch.float64), tmp_path
    )
    assert metrics["reference_energy"] == HookeSingletExact.exact_energy
    assert metrics["energy_abs_error"] == abs(metrics["energy_error"])


def test_the_stratified_sample_count_can_support_a_variance() -> None:
    """1024 points, not the 8 and 4 the invariant tasks use.

    ``eval.yaml:207,229`` uses 8 and 4 for permutation-consistency checks, where
    each point need only agree with its own image. A variance needs far more: a
    quarter of the stratum weight sits on ``cusp`` with ``r12_min = 1.0e-4``,
    where E_L is near-singular and the sample statistics are heavy-tailed, so at
    8 points the number tracks whichever run drew the worst outlier.
    """

    generator = _config("eval").evaluation_tasks.stratified_geometry.generator

    assert generator.n_samples >= 1024
    assert set(generator.strata) == {"cusp", "moderate_pair", "bulk", "tail"}
    assert generator.bounds.cusp.r12_min == pytest.approx(1.0e-4)


@pytest.mark.parametrize(
    ("basis_slot", "activation_slot", "channels", "lr"),
    tuple(
        zip(
            BASIS_LEVELS,
            ACTIVATION_LEVELS,
            CHANNELS_LEVELS + CHANNELS_LEVELS[:1],
            LR_LEVELS + LR_LEVELS,
            strict=True,
        )
    ),
)
def test_the_primary_metric_sampling_budget_is_identical_for_every_grid_row(
    basis_slot: str, activation_slot: str, channels: int, lr: float
) -> None:
    """No axis reaches the validation sampler's budget.

    The primary selection metric is a Monte Carlo mean. If its walker count,
    burn-in or decorrelation length moved with the configuration being scored,
    two rows' means would carry different amounts of sampling noise and the
    comparison -- which is the study's deliverable -- would not be a comparison
    of wavefunctions. Only the seed is allowed to differ.
    """

    budget_keys = ("n_walkers", "burn_in", "n_steps", "proposal_scale")
    reference = _config("eval")
    row = _config(
        "eval",
        basis_slot=basis_slot,
        activation_slot=activation_slot,
        channels=channels,
        lr=lr,
    )

    for key in budget_keys:
        assert row.evaluation_sampler[key] == reference.evaluation_sampler[key]
        assert _raw_text("eval", f"evaluation_sampler.{key}").startswith("${evaluation_sampler_params.")
        assert isinstance(reference.evaluation_sampler_params[key], (int, float))
    assert row.evaluation_tasks.mcmc_energy.generator.max_samples == reference.evaluation_sampler.n_walkers


def test_every_evaluation_task_is_registered_with_the_evaluator() -> None:
    """A defined-but-unregistered task is a metric that silently never appears."""

    cfg = _config("eval")
    registered = [task.name for task in cfg.evaluator.tasks]

    assert registered[:2] == ["mcmc_energy", "stratified_geometry"]
    assert set(registered) == set(cfg.evaluation_tasks.keys())


# ---------------------------------------------------------------------------
# Watch item: channels rescales psi at initialization
# ---------------------------------------------------------------------------


def _fixed_batch(n_samples: int = 256) -> ElectronBatch:
    """Return one deterministic two-electron batch, identical at every width."""

    generator = torch.Generator().manual_seed(20260813)
    positions = torch.randn(n_samples, 2, 3, dtype=torch.float64, generator=generator)
    spins = torch.tensor([[1.0, -1.0]], dtype=torch.float64).repeat(n_samples, 1)
    return ElectronBatch(positions=positions, spins=spins)


@pytest.mark.parametrize("channels", CHANNELS_LEVELS)
def test_pfaffian_readout_weights_are_one_over_channels_at_every_width(channels: int) -> None:
    """The mechanism of the channels/effective-lr confound, pinned exactly.

    ``PfaffianReadout`` defaults ``trainable: false`` (``readout/pfaffian.py:102``)
    and holds fixed weights ``1 / pair_channels`` (``:111,117``), and
    ``Psi = sum_c w_c Pf[K_c]``. So the initial amplitude scale carries an explicit
    ``1/C`` factor: widening the axis rescales psi at initialization and therefore
    shifts the effective step size on the readout path, which two lr levels do not
    cover. Recorded here, not fixed -- the scan report must account for it rather
    than attribute the whole channels effect to capacity.
    """

    readout = instantiate(_config("train", channels=channels).model.readout)

    weights = readout.channel_weight_buffer

    assert readout.trainable is False
    assert readout.pair_channels == channels
    torch.testing.assert_close(weights, torch.full((channels,), 1.0 / channels, dtype=weights.dtype))


def test_initial_logabs_scale_is_recorded_for_every_channels_level(
    record_property: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Measure and record log|psi| at initialization for channels 8 / 16 / 32.

    The confound this documents: with fixed ``1/C`` readout weights the initial
    amplitude carries a ``1/C`` factor, so log|psi| is displaced by roughly
    ``-log C`` plus whatever the per-channel kernel scale contributes. The
    measured values are attached to the test record so the scan report can quote
    them instead of re-deriving them.

    The assertion is deliberately weak -- finite, and demonstrably width
    dependent. A tight numeric band would encode this torch build's arithmetic,
    and the point is to make the confound visible, not to freeze it.
    """

    batch = _fixed_batch()
    measured: dict[int, float] = {}
    for channels in CHANNELS_LEVELS:
        torch.manual_seed(20260813)
        # `runtime.dtype: float64` is applied by the runner, not by `instantiate`,
        # so the cast is explicit here to match the batch.
        model = instantiate(_config("train", channels=channels).model).to(dtype=torch.float64)
        with torch.no_grad():
            logabs = model(batch).logabs
        assert torch.isfinite(logabs).all(), f"channels={channels} produced nonfinite log|psi|"
        measured[channels] = float(logabs.mean())
        record_property(f"initial_logabs_mean_channels_{channels}", measured[channels])

    with capsys.disabled():
        print("\ninitial log|psi| mean by channels level (fixed 256-point batch, seed 20260813):")
        for channels, value in measured.items():
            print(f"  channels={channels:>2}  mean_logabs={value:+.6f}  implied -log C = {-math.log(channels):+.6f}")

    assert len(set(measured.values())) == len(CHANNELS_LEVELS), (
        "initial log|psi| did not move with channels; the 1/C readout rescaling "
        f"should displace it, measured {measured}"
    )
