"""The closed helium-importance train schema.

This module owns the *policy*: which sections a helium-importance training
configuration may declare, which key families it may never contain, and which
callbacks it may install. The sweep mechanism that enforces it lives in
:mod:`tpen.config_schema`.

Declaring the schema
--------------------
The firewall is selected by a top-level ``schema`` key holding
:data:`HI_TRAIN_SCHEMA`::

    schema: tpen.hi.train.v1

For a helium-importance configuration that key is MANDATORY, not optional, and
omitting it is a loud refusal rather than a quiet pass. A closed schema that
applies only when a config remembers to ask for it is not closed -- it is
closed-if-you-remember, and the omission is silent and permanent.

Membership in the family is decided POSITIVELY, by ``experiment.name``, rather
than by defaulting the firewall on for every configuration in the repository
and exempting the rest. See the comment on :data:`HI_EXPERIMENT_NAME` for the
measurement behind that choice, and for the residual hole it leaves.

A configuration outside the family passes through untouched. That is how the
frozen historical fixtures survive without an exemption list:
``experiments/atomistic/he-v1/configs/train.yaml`` carries a
``reference_energy`` in its ``system`` block and the plan of record designates
it preserved rather than edited -- and it is ``tpen_he_v1``, so it is simply not
this family.

Why a training configuration may not hold a reference
-----------------------------------------------------
A reference energy reachable during training is an arm-selection hazard: any
quantity that can be compared against it mid-run can be used, deliberately or
not, to choose between arms on accuracy. The separately loaded evaluation
manifest is the only permitted reference holder, so the comparison can only
happen after the training configuration is frozen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tpen.config_schema import (
    ClosedSchemaError,
    canonical_digest,
    ForbiddenSurface,
    Rejection,
    SchemaPolicy,
    iter_nodes,
    sweep_environment,
    sweep_raw,
    sweep_resolved,
)

__all__ = [
    "ADMITTED_CALLBACK_TARGETS",
    "ADMITTED_METHOD_TARGETS",
    "ADMITTED_UPDATE_METHOD_TARGETS",
    "REFERENCE_MANIFEST_MODULE",
    "HI_METHOD_ROSTER",
    "MethodAvailability",
    "HI_TRAIN_POLICY",
    "HI_EXPERIMENT_NAME",
    "HI_TRAIN_SCHEMA",
    "SCHEMA_KEY",
    "canonical_train_identity",
    "declared_schema",
    "is_hi_family",
    "validate_hi_train_config",
]


# Durable external identifier: it is written into config files, so it is not
# renameable without rewriting every configuration that opts in.
HI_TRAIN_SCHEMA = "tpen.hi.train.v1"

# Top-level key a configuration uses to select its schema.
SCHEMA_KEY = "schema"

# ---------------------------------------------------------------------------
# Why declaring the schema is MANDATORY for this family, not optional
# ---------------------------------------------------------------------------
# A closed schema enforced only when a configuration remembers to ask for it is
# not closed; it is closed-if-you-remember, and the failure is silent and
# permanent. A new helium-importance train config that omits ``schema:`` would
# get ZERO enforcement and nothing anywhere would go red.
#
# So the marker is required of every configuration that IDENTIFIES ITSELF as
# helium-importance, and its absence is a loud refusal.
#
# The family is detected POSITIVELY, by experiment name, rather than by
# defaulting the firewall on for everything and exempting the rest. MEASURED
# across the repository's configs, the experiment names present are
# tpen_he_importance, tpen_he_v1, tpen_h2_v1, tpen_pair_v1, and four hooke
# configs whose name is the interpolation ``${study.name}``. Defaulting on
# would therefore refuse three other lanes' experiments, and the "narrow
# exemption list" needed to rescue them would be a register of everything that
# is NOT helium-importance -- unbounded, and growing with every new experiment
# anywhere in the repository. A positive detector grows with THIS study
# instead, and needs no exemption entry for the frozen he-v1 fixture at all,
# because that fixture is ``tpen_he_v1`` and simply is not in the family.
#
# RESIDUAL HOLE, stated rather than papered over: a configuration that omits
# BOTH the schema key AND the helium-importance experiment name is not caught
# here. That is evasion, not omission, and omission is the failure this rule
# exists to close. The repo-level test in
# ``tests/unit/test_hi_reference_separation.py`` is the second, independent net
# for the static case; neither net alone covers the other's population, which
# is why there are two.
HI_EXPERIMENT_NAME = "tpen_he_importance"


# ---------------------------------------------------------------------------
# Forbidden key families
# ---------------------------------------------------------------------------
# Each family names a class of value that must not be reachable from a training
# configuration. Token sets are deliberately narrow: see the token-versus-
# substring discussion in :mod:`tpen.config_schema` for why a wider set would
# reject ordinary words while still looking like a working firewall.

REFERENCE_SURFACE = ForbiddenSurface(
    name="reference",
    tokens=frozenset({"reference", "baseline", "truth"}),
    reason=(
        "a training configuration may not hold or reach a reference value; the "
        "separately loaded evaluation manifest is the only permitted reference holder"
    ),
)

GAP_SURFACE = ForbiddenSurface(
    name="gap",
    tokens=frozenset({"gap", "gaps"}),
    reason=(
        "an energy gap is a derived reference quantity and belongs to evaluation, "
        "not to a training configuration"
    ),
)

BAND_SURFACE = ForbiddenSurface(
    name="band",
    tokens=frozenset({"band", "bands"}),
    reason=(
        "an accuracy band encodes a selection rule; selection belongs to the "
        "confirmation lane, not to a training configuration"
    ),
)

CONTINUATION_SURFACE = ForbiddenSurface(
    name="continuation",
    tokens=frozenset({"continuation", "continue"}),
    reason=(
        "the continuation ladder is retired in favour of train/eval/test; a "
        "decide-then-extend surface must not be configurable here. Note that "
        "checkpoint RESUME is unaffected and remains permitted -- resuming an "
        "interrupted run is recovery, whereas continuation is selection"
    ),
)
# DELIBERATELY BROADER THAN "target-triggered". The surface this closes is a
# stop rule keyed to a target, but the tokens refuse ANY configurable stop rule,
# because a rule cannot be inspected for what it is keyed to at schema time:
# ``trainer.stop_at: -2.9037`` names no reference yet is target-triggered, while
# ``trainer.stop_at_step: 50000`` is a budget. Refusing the family and requiring
# a schema change to re-admit one is the narrow-token doctrine applied to a
# surface whose hazard lives in the VALUE rather than the key.
#
# The over-restriction risk was measured, not assumed: no configuration under
# ``experiments/atomistic/he-importance/configs/`` declares a key carrying any
# of these tokens, and ``TestEveryHIConfigDeclaresTheSchema`` validates all of
# them, so an over-wide token set here turns the existing suite red rather than
# surfacing later as a run that cannot start.
#
# ``early_stopping`` and ``early_stop`` are both caught -- ``stopping`` and
# ``stop`` are separate tokens, so neither spelling needs its own entry.
STOP_RULE_SURFACE = ForbiddenSurface(
    name="stop-rule",
    tokens=frozenset({"stop", "stopping", "halt", "patience"}),
    reason=(
        "a training configuration may not declare a stop rule; a run that halts "
        "when it reaches a target has read the target, and a stop rule is "
        "arm-selection performed inside training rather than in the "
        "confirmation lane. Every arm runs its declared budget"
    ),
)


# The evaluation manifest's module, named as a STRING and deliberately NOT
# imported. Importing it here would put ``tpen.hi_manifest`` on the training
# path as a DIRECT import, which is exactly what
# ``tests/unit/test_hi_reference_separation.py`` censuses -- so the firewall
# would breach the invariant it enforces. Said as "direct import" rather than
# "reachability" on purpose: that test is a direct-import census and does not
# establish reachability in general.
# That test carries its own copy of this name and a control asserts the two
# agree, because a silent divergence would leave the rule below pointing at a
# module nobody loads.
REFERENCE_MANIFEST_MODULE = "tpen.hi_manifest"


# ---------------------------------------------------------------------------
# Closed section set
# ---------------------------------------------------------------------------
HI_TRAIN_SECTIONS = frozenset(
    {
        SCHEMA_KEY,
        "experiment",
        "run",
        "runtime",
        "system",
        "atoms",
        "runner",
        "model",
        "hamiltonian_terms",
        "sampler",
        "optimizer",
        "trainer",
        "callbacks",
        "loggers",
    }
)


# ---------------------------------------------------------------------------
# Admitted callbacks
# ---------------------------------------------------------------------------
# An ALLOWLIST rather than a denylist, because the hazard here is a callback
# nobody thought to forbid. A denylist admits every callback written after it,
# including a future in-training evaluation callback that reports an energy
# against a reference -- exactly the surface this schema exists to close.
#
# Membership means "carries no reference and reports no energy". Notably absent:
# any diagnostic that accepts a ``reference_energy`` (see
# ``tpen/diagnostics/energy.py``), and any in-training evaluation callback,
# which the plan of record defers pending a separate approval of its second
# RNG / dual payload / lag confound.
ADMITTED_CALLBACK_TARGETS = frozenset(
    {
        # Provenance and run bookkeeping.
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.Status",
        "tpen.callback.ArtifactIndex",
        "tpen.callback.FailureLog",
        # Correctness and health guards. None reports an energy value.
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.SamplerHealth",
        "tpen.callback.FactorScalars",
        "tpen.callback.RuntimeEquivariance",
        # Durable state and resource accounting.
        "tpen.callback.Checkpoint",
        "tpen.callback.ResourceUsage",
        # Timing observability.
        "tpen.callback.RunTiming",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.DiagnosticTiming",
    }
)


# ---------------------------------------------------------------------------
# Admitted method coverage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodAvailability:
    """One optimizer method in the study roster, and whether it may run yet.

    Parameters
    ----------
    method : str
        The study's name for the method, e.g. ``"sr"``. This is the vocabulary
        the materializer uses when it marks a cell unavailable.
    admitted : bool
        Whether a training configuration may name this method today.
    target : str or None
        The config ``_target_`` that selects it, when admitted. ``None`` whenever
        the method may not be selected, which covers two different situations
        that must not be conflated:

        - no implementation exists, and inventing a module path would make this
          table claim otherwise;
        - an implementation exists but is EXCLUDED from this study, so naming it
          would invite someone to select it.

        ``requires`` is what distinguishes them, so it has to stay accurate as
        the repository moves underneath it. SR is the live example: its
        implementation landed while this table said it was waiting for that
        implementation.
    requires : str
        What admission is waiting on. Reported verbatim when a configuration
        names the method, so the refusal says what would change it.
    """

    method: str
    admitted: bool
    target: str | None
    requires: str


# The roster is the five methods of the optimizer grid. Only Adam is admitted
# today; the other four are UNAVAILABLE, which is a different thing from absent.
# An unavailable cell must stay visibly unavailable rather than quietly become
# Adam -- a silent substitution would make a run claim it compared a method it
# never ran.
HI_METHOD_ROSTER: tuple[MethodAvailability, ...] = (
    MethodAvailability(
        method="adam",
        admitted=True,
        target="torch.optim.Adam",
        requires="admitted; the first scientific roster runs on Adam",
    ),
    MethodAvailability(
        method="sr",
        admitted=False,
        target=None,
        # EXCLUDED FROM THIS STUDY, not awaiting implementation. SR/minSR LANDED
        # on dev (Lane N, PRs #472 and #476), so `tpen/training/sr.py` exists and
        # works -- and it still cannot be a scan arm here, because it does not
        # run on two-electron models and helium is two electrons.
        #
        # The distinction matters and is why this text is not "waiting for Lane
        # N": that phrasing was true until Lane N merged, and would now send a
        # reader to satisfy a prerequisite that is ALREADY satisfied, from which
        # they would conclude the refusal is a bug. A refusal that survives its
        # stated reason misdirects harder than one with no reason at all.
        requires=(
            "EXCLUDED from the helium-importance scan on scientific grounds: SR/minSR is "
            "implemented and merged (Lane N), but does not run on two-electron models and "
            "helium is two electrons. This is not pending work and no amount of optimizer "
            "work will admit it; admitting it would need a different system"
        ),
    ),
    MethodAvailability(
        method="kfac",
        admitted=False,
        target=None,
        requires="a passing fail-closed kfac-pytorch compatibility gate; a failing gate leaves KFAC unavailable",
    ),
    MethodAvailability(
        method="spring",
        admitted=False,
        target=None,
        requires="admitted minSR first, then the thin projected-history recurrence above it",
    ),
    MethodAvailability(
        method="linear_method",
        admitted=False,
        target=None,
        requires="a separately authorized Hamiltonian-tangent design gate and dense-memory admission",
    ),
)

# SCOPED TO ``optimizer._target_``, and to nothing else. The roster answers
# "which optimizer may a cell name", so this set is the only thing
# `_sweep_method` compares against -- and for a long while it was the only
# method qualification in the schema at all, which read as though naming a
# method anywhere was covered. It is not: the UPDATE RULE is selected by
# ``trainer.update_method``, an independent surface with its own allowlist
# below. A reader who meets this set should meet that fact here rather than
# discover it from a run.
ADMITTED_METHOD_TARGETS = frozenset(
    entry.target for entry in HI_METHOD_ROSTER if entry.admitted and entry.target
)


# ---------------------------------------------------------------------------
# Admitted update methods
# ---------------------------------------------------------------------------
# THE SECOND HALF OF METHOD ADMISSION. `VMCTrainer` takes an `update_method`
# spec, and `_select_update_method` resolves a Hydra ``_partial_`` block into
# the object that performs every parameter update. So a configuration selects
# its update RULE here and its optimizer in `optimizer` -- two surfaces, and
# only the second was qualified. `tpen.training.sr.StochasticReconfigurationUpdate`
# named here was admitted unqualified even though the roster records SR as
# EXCLUDED from this study.
#
# The observed example is a PRECONSTRUCTION gap rather than a successful
# unadmitted run: SR with Adam is refused later by the SR constructor. That is
# what makes this lower severity than the other holes, and it is not a reason to
# leave it open -- "some other component happens to refuse it" is a property of
# today's constructors, not a rule, and the schema's job is to refuse before
# anything is constructed.
#
# ABSENCE IS ADMITTED, and it is what the control config does. `update_method:
# null` (or omission) makes `_select_update_method` build `LegacyAutogradUpdate`
# -- the plain optimizer step, which IS the admitted Adam method. Requiring the
# declaration would refuse every shipped configuration.
#
# BOTH SPELLINGS of the admitted class are listed. `tpen.training.__init__`
# re-exports it, so Hydra resolves either path to the same object, and an
# allowlist that named one would refuse a configuration that is correct. A
# FULL-PATH set rather than a trailing-component one, matching
# :data:`ADMITTED_CALLBACK_TARGETS`: both are sets of executable targets, and a
# trailing-component match would admit any class anywhere that happened to
# share the name.
ADMITTED_UPDATE_METHOD_TARGETS = frozenset(
    {
        "tpen.training.update.LegacyAutogradUpdate",
        "tpen.training.LegacyAutogradUpdate",
    }
)

# Adam coordinates that §2.7 fixes for every cell. ``lr`` and ``beta2`` are
# deliberately absent: those are the SCAN coordinates (four and two levels
# respectively), and pinning them here would make the study's own grid
# unrunnable. This table holds only what no arm may vary.
_FIXED_ADAM_COORDINATES: dict[str, object] = {
    "eps": 1e-8,
    "weight_decay": 0,
}

# The first Adam moment is fixed at .9; only the second is scanned.
_FIXED_ADAM_BETA1 = 0.9


HI_TRAIN_POLICY = SchemaPolicy(
    name=HI_TRAIN_SCHEMA,
    forbidden_surfaces=(
        REFERENCE_SURFACE,
        GAP_SURFACE,
        BAND_SURFACE,
        CONTINUATION_SURFACE,
        STOP_RULE_SURFACE,
    ),
    allowed_sections=HI_TRAIN_SECTIONS,
    # ``oc.env`` reads the process environment and ``now`` reads the clock.
    # Either makes the resolved configuration a function of something other
    # than the file and its overrides, so two ranks could resolve the same file
    # differently. ``oc.decode`` is included because it evaluates arbitrary text
    # that may itself be built from environment interpolation.
    #
    # KEPT FOR ITS MESSAGES, NOT FOR ITS COVERAGE. The allowlist below already
    # refuses all four; naming them keeps a config that reaches for the obvious
    # one getting the specific reason rather than the generic refusal.
    forbidden_resolvers=frozenset({"oc.env", "env", "now", "oc.decode"}),
    # THE CLOSED SET OF RESOLVER CALLS, and it is EMPTY.
    #
    # A DENYLIST WAS BEATEN TWICE HERE. First by nesting: a forbidden resolver
    # inside a permitted one was invisible to a regex that stopped at the first
    # closing brace. Then by an unnamed sibling: ``oc.decode`` was refused by
    # NAME for evaluating arbitrary text, and ``oc.create`` does the same job
    # under a different name AND re-resolves interpolations in its output.
    # MEASURED at the previous head: a config carrying ``oc.create`` around a
    # YAML string whose unicode escapes hid the interpolation opener from the
    # raw sweep resolved ``run.run_id`` to ``rank_7`` from ``SLURM_PROCID``.
    # Every rank would resolve a different identity while the rank-divergence
    # rule passed, because the id was non-null, and the canonical digest would
    # report a difference it could not explain.
    #
    # That is this slice's own class applied to the resolver rule: a rule that
    # reads NAMES defeated by something supplying the same capability without
    # that name. The set of text-evaluating resolvers is open-ended and grows
    # with OmegaConf. The set this study needs is not.
    #
    # EMPTY IS MEASURED, NOT ASPIRATIONAL: the shipped control config contains
    # 32 plain node references and ZERO resolver calls, so this refuses nothing
    # that exists. Node references such as ``${system.spatial_dim}`` are
    # untouched -- they name configuration, not a resolver. Admitting one later
    # is a schema change, which is the point: it becomes visible in review.
    allowed_resolvers=frozenset(),
)


def declared_schema(cfg: Any) -> str | None:
    """Return the schema a configuration opts in to, if any.

    Parameters
    ----------
    cfg : Any
        A ``DictConfig`` or plain mapping.

    Returns
    -------
    str or None
        The value of the top-level ``schema`` key, or ``None`` when the
        configuration declares none.
    """

    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, SCHEMA_KEY, default=None)
    elif isinstance(cfg, Mapping):
        value = cfg.get(SCHEMA_KEY)
    else:
        return None
    return None if value is None else str(value)


def is_hi_family(cfg: Any) -> bool:
    """Return whether a configuration identifies itself as helium-importance.

    Parameters
    ----------
    cfg : Any
        A ``DictConfig`` or plain mapping.

    Returns
    -------
    bool
        ``True`` when ``experiment.name`` is :data:`HI_EXPERIMENT_NAME`.

    Notes
    -----
    Read WITHOUT resolving, so a config whose interpolations are broken is
    still recognised as belonging to the family and still refused for omitting
    the schema key. Resolving here would make an unrelated typo silently
    downgrade a helium-importance config to an unenforced one -- the finding
    and the thing that hides it would share a failure domain.
    """

    if isinstance(cfg, DictConfig):
        try:
            name = OmegaConf.select(cfg, "experiment.name", default=None)
        except Exception:  # noqa: BLE001 - a broken tree must not grant an exemption
            return False
    elif isinstance(cfg, Mapping):
        experiment = cfg.get("experiment")
        name = experiment.get("name") if isinstance(experiment, Mapping) else None
    else:
        return False
    return name == HI_EXPERIMENT_NAME


def _sweep_callbacks(resolved_tree: Any) -> list[Rejection]:
    """Reject any configured callback outside the admitted set.

    Checked on the resolved tree only. A callback's ``_target_`` may itself be
    an interpolation, in which case the raw tree holds ``"${...}"`` rather than
    a class path and has nothing to compare against the allowlist.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    callbacks = resolved_tree.get("callbacks")
    if callbacks is None:
        return []

    rejections: list[Rejection] = []
    for path, key, value in iter_nodes({"callbacks": callbacks}):
        if key != "_target_" or not isinstance(value, str):
            continue
        # Only the callback's own ``_target_`` is a callback identity; a nested
        # ``_target_`` is a constructor argument (a schedule, a payload, a
        # probe) and is governed by its owning callback, not by this allowlist.
        # THE BOUNDARY IS UNCHANGED AND THE REASON STILL HOLDS. What changed is
        # that a nested target is no longer ungoverned: `_sweep_target_values`
        # refuses an executable target that names the evaluation reference,
        # anywhere in the tree and at any depth. See its docstring for why that
        # is a separate rule rather than a wider allowlist here.
        owner = path.rsplit("._target_", 1)[0]
        if owner.count(".") != 0 or not owner.startswith("callbacks["):
            continue
        if value not in ADMITTED_CALLBACK_TARGETS:
            rejections.append(
                Rejection(
                    rule="unadmitted-callback",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"callback {value!r} is not in the admitted set. Admission means the "
                        "callback carries no reference and reports no energy. Add it to "
                        "tpen.hi_schema.ADMITTED_CALLBACK_TARGETS only after confirming that"
                    ),
                )
            )
    return rejections


def _sweep_target_values(resolved_tree: Any) -> list[Rejection]:
    """Reject an executable ``_target_`` that names the evaluation reference.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per offending target, at any depth in any section.

    Notes
    -----
    WHAT THIS CLOSES, MEASURED. :meth:`ForbiddenSurface.matches` tests KEYS, so
    a value is never tokenized. Injecting
    ``_target_: tpen.hi_manifest.reference_energy`` at eight points in the
    control config was ACCEPTED at six of them -- ``loggers``, ``sampler``,
    ``model``, ``runner``, ``trainer`` and the nested ``model.embedding``. Only
    ``callbacks`` and ``optimizer`` refused it, and each of those refuses for
    its own reason (an allowlist, and the method roster) rather than because
    targets were governed. **Five root sections had no target rule at all**, so
    the nested-callback-target finding this rule was filed under was one
    instance of the gap rather than the gap itself. With this rule the same
    eight injections are refused eight of eight, and the six that changed are
    refused by NAME here rather than incidentally by some other rule -- see
    ``TestNoTargetCanNameTheReferenceModule``, which asserts the rule name and
    not merely that something was refused.

    WHY A DENYLIST HERE AND AN ALLOWLIST FOR CALLBACKS. The callback allowlist
    is enumerable: the study installs a fixed set of bookkeeping and health
    callbacks and a new one is a review event. The targets in ``model``,
    ``sampler`` and ``trainer`` are NOT enumerable at schema time -- the scan
    varies producers, activations, update rules and five initializations, so an
    allowlist would have to list every arm the materializer may emit and would
    refuse a legitimate arm the day one is added. That is the over-restriction
    Amendment A warns about, and it surfaces as a run that cannot start rather
    than as a red test. The hazard being closed is narrow and nameable -- an
    executable that loads the evaluation reference -- so it is named.

    TWO RULES, AND NEITHER SUBSUMES THE OTHER. The module rule catches
    ``tpen.hi_manifest.load_evaluation_manifest``, whose tokens are
    ``tpen, hi, manifest, load, evaluation, manifest`` and contain no forbidden
    token at all. The token rule catches
    ``tpen.diagnostics.energy.ReferenceGapProbe``, which lives outside the
    manifest module entirely. Each was checked against the other's example.

    RESOLVED TREE ONLY, for the same reason :func:`_sweep_callbacks` gives: a
    ``_target_`` may itself be an interpolation, and the raw tree then holds
    ``"${...}"`` with nothing to compare. The key-side sweeps run on both trees
    already, so a forbidden KEY is still caught in raw.

    RESIDUAL, stated rather than papered over, and CORRECTED after a reviewer
    read the other net instead of taking this docstring's word for it. This
    refuses targets that NAME the reference. A target that reaches the reference
    TRANSITIVELY -- a module importing a helper that imports the manifest -- is
    not visible here.

    An earlier version of this paragraph said that case "is what the import
    separation test covers". **That was wrong, and it was wrong in the
    comfortable direction.** ``tests/unit/test_hi_reference_separation.py`` is a
    DIRECT-IMPORT CENSUS: its own docstring records three static shapes that
    reach the holder and are invisible to it, including a transitive chain
    through a helper outside the swept roots. So transitive reachability is
    covered by NEITHER net, not by the other one, and writing that it was
    covered turned a real gap into a division of labour that does not exist.

    What IS covered between the two: a target naming the reference module or
    carrying a forbidden token (here), a direct import from a swept module
    (there), and a diagnostic that ACCEPTS a reference, which needs a
    ``reference_energy`` argument key that the key-side sweep refuses.
    """

    rejections: list[Rejection] = []
    for path, key, value in iter_nodes(resolved_tree):
        if not isinstance(value, str):
            continue
        target = value.strip()
        # THE MODULE-IDENTITY CHECK APPLIES TO EVERY STRING VALUE, not only to
        # ``_target_``. A generic importer carries the module path as DATA under
        # a key nobody tokenizes:
        #
        #     {_target_: importlib.import_module, name: tpen.hi_manifest}
        #     {_target_: hydra.utils.get_method, path: tpen.hi_manifest.load_...}
        #
        # MEASURED: both VALIDATED in a free-form runner slot while the direct
        # spelling in the same slot was refused, and recursive instantiation
        # imports the reference-holder module onto the training path.
        #
        # ENUMERATING THE IMPORTERS WOULD BE THE SAME MISTAKE AGAIN --
        # ``get_class``, ``get_object``, ``pydoc.locate``, ``pkgutil.resolve_name``
        # and whatever is added next. So the module PATH is what is checked
        # rather than the importer.
        #
        # WHAT THIS DOES NOT REACH, stated because the first version of this
        # comment claimed more. It catches the DOTTED spelling wherever it
        # appears. It does NOT catch a path split across two arguments
        # (``name=".hi_manifest", package="tpen"``) or written in filesystem
        # form (``runpy.run_path("tpen/hi_manifest.py")``), both MEASURED
        # validating. "The path must appear somewhere as a string" is true only
        # if it appears WHOLE.
        #
        # Safe to widen because this is EXACT MODULE IDENTITY, not a token: it
        # matches ``tpen.hi_manifest`` and its submodules and nothing else. The
        # TOKEN check below stays scoped to ``_target_``, where the value names
        # code that will run -- widening THAT to every value would refuse a run
        # directory named ``outputs/baseline``.
        #
        # IMPACT BOUND, RETRACTED. This paragraph used to add that the
        # reference NUMBERS were safe because "no config-only shape can CALL the
        # loader". **That was false, and it was the class error this module is
        # about, committed in a bound rather than in a rule**: it bounded a
        # hazard by ONE LOADER'S NAME. An independent verifier measured
        # ``{_target_: omegaconf.OmegaConf.load, file_: <the manifest path>}``
        # VALIDATING and returning ``reference.energy ==
        # -2.9037243770341195``. No module path, no forbidden token -- the
        # manifest path tokenizes to nothing this schema denies.
        #
        # So the honest bound is only this: this RULE refuses the dotted module
        # spelling wherever it appears as a string. It does not bound what a
        # configuration can read. Tracked as its own item; see the residual list
        # on :func:`_sweep_positional_construction`.
        if target == REFERENCE_MANIFEST_MODULE or target.startswith(
            f"{REFERENCE_MANIFEST_MODULE}."
        ):
            rejections.append(
                Rejection(
                    rule="forbidden-target:reference-module",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"{target!r} names {REFERENCE_MANIFEST_MODULE!r}, which holds the "
                        "evaluation reference. Importing or instantiating it would load the "
                        "reference onto the training path, which is what this schema exists to "
                        "prevent. Checked on EVERY string value, not only on _target_: a "
                        "generic importer such as importlib.import_module or "
                        "hydra.utils.get_method carries the module path as an argument, so a "
                        "rule reading only _target_ sees an innocuous importer. The reference "
                        "is read after training, by a separate process"
                    ),
                )
            )
            continue
        if key != "_target_":
            continue
        for surface in HI_TRAIN_POLICY.forbidden_surfaces:
            if surface.matches(target):
                rejections.append(
                    Rejection(
                        rule=f"forbidden-target:{surface.name}",
                        tree="resolved",
                        path=path,
                        detail=(
                            f"_target_ {target!r} names a {surface.name} surface; "
                            f"{surface.reason}. A _target_ is executable, so the token check "
                            "applies to the value here and not only to the key"
                        ),
                    )
                )
    return rejections


# ---------------------------------------------------------------------------
# Frozen architectural coordinates
# ---------------------------------------------------------------------------
# Only coordinates the study does NOT vary appear here. The scan varies the
# producer/path policy, channels, activations, embedding width/depth, the
# feature update rule, and five initializations; pinning any of those would
# make the study's own grid unrunnable. Every key name below was checked
# against experiments/atomistic/he-v1/configs/train.yaml, a real production
# config -- a constraint naming a key that no config spells would never fire
# and would be a silent no-op rather than a check.

# Value must equal the expected one wherever the key appears anywhere under
# ``model``. Depth-independent because the producer/path policy varies the
# nesting: A1/A2 swap tensor, linear and hybrid producers, so a fixed path
# would stop matching on some arms.
_FROZEN_MODEL_KEYS: dict[str, tuple[object, str]] = {
    "max_order": (2, "body order 2 (literal control, stack/path metadata)"),
    "max_virtual_order": (2, "virtual support fixed at 2, no V1 arm (A3)"),
    "implementation": (
        "vectorized",
        "vectorized kernels; the slow implementation is an oracle, not a scientific arm",
    ),
}

# Singular coordinates, checked BY KEY AT ANY DEPTH rather than at a dotted
# path, and value-sensitively.
#
# THE DOTTED VERSION CHECKED WHERE THE VALUE IS DECLARED, NOT WHERE IT IS
# CONSUMED. `system.spatial_dim` and `runtime.dtype` are the study's canonical
# declarations, but the components are built from `model.embedding.spatial_dim`,
# `sampler.spatial_dim` and `sampler.dtype`. The shipped config wires those with
# interpolations -- and nothing required it to. MEASURED: with
# `system.spatial_dim: 3` left compliant, `model.embedding.spatial_dim: 2`,
# `sampler.spatial_dim: 2` and `sampler.dtype: float32` ALL VALIDATED, and the
# model would be built in two dimensions.
#
# That is F5's own shape -- the schema validating ROOT fields while what is
# actually constructed carries its own copy -- applied to a scalar instead of a
# component. It is the finding this slice exists for, one more time.
#
# ABSENT IS FINE, matching the previous behaviour: absence means the code's own
# default applies, and these three defaults are correct. Only a divergent VALUE
# is refused, wherever it appears.
#
# OVER-RESTRICTION MEASURED, not assumed: in the shipped configuration every
# `spatial_dim` node resolves to 3 and every `dtype` node to 'float64', so this
# refuses nothing that exists. A future component using `dtype` to mean
# something else would be refused LOUDLY, and the remedy would be to name the
# coordinate rather than to re-anchor the rule.
_FROZEN_SCALAR_KEYS: dict[str, tuple[object, str]] = {
    "spatial_dim": (3, "spatial dimension 3 (literal control, system/numerics)"),
    "dtype": ("float64", "float64 (literal control, system/numerics)"),
}

# Hamiltonian-term coordinates the study does NOT vary, keyed by the trailing
# component of the term's ``_target_`` and then by argument name.
#
# BY SHAPE, NOT BY PATH, and this replaces a dotted
# ``hamiltonian_terms.electron_nucleus.eps`` entry in the table above that was
# ESCAPABLE with ordinary keyword config. ``normalize_hamiltonian_terms``
# accepts a Mapping OR a Sequence -- a sequence falls back to snake-case class
# names -- and the mapping's keys are the author's choice, not the schema's. So
# both of these construct the same Hamiltonian while defeating a dotted path:
#
#     hamiltonian_terms:            hamiltonian_terms:
#       - _target_: ...Kinetic        en:
#       - _target_: ...ElectronNucleus  _target_: ...ElectronNucleusPotential
#         eps: 0.01                     eps: 0.01
#
# MEASURED: both validated with ``eps: 0.01`` while the mapping form spelled
# ``electron_nucleus`` was correctly refused. The floor matters -- a nonzero one
# masks the near-nucleus cancellation the local-energy qualification has to
# measure -- so the rule now identifies the term by WHAT IT IS.
#
# ABSENT IS ADMISSIBLE: ``ElectronNucleusPotential.__init__`` defaults ``eps``
# to 0.0, so omitting it applies the value the study wants. Only a divergent
# VALUE is refused, matching the frozen-scalar precedent above.
#
# DELIBERATELY NOT EXTENDED to the electron-electron floor. That term also
# takes ``eps`` and also defaults to 0.0, but no slice has pinned it and its own
# docstring records the finite-eps electron-electron case as UNMEASURED.
# Pinning it here would be a new scientific constraint smuggled in under a
# bug fix; it is FILED rather than added.
_FROZEN_TERM_COORDINATES: dict[str, dict[str, tuple[object, str]]] = {
    "ElectronNucleusPotential": {
        "eps": (
            0.0,
            "Coulomb distance floor 0.0; a floor would mask near-nucleus cancellation",
        ),
    },
}

# Trainability must be DECLARED, never inherited. Absence is a violation here,
# unlike the frozen scalars above, because every one of these defaults to the
# WRONG value. This is not hypothetical: he-v1's own config records that
# PfaffianReadout defaults to trainable=False, so passing only `channels`
# pinned the channel weights at a uniform 1/32 for all 300,000 updates with
# nothing in named_parameters(), nothing in state_dict(), and no log line. The
# silence was total. Requiring the declaration is what makes that failure
# impossible to repeat by omission.
# Readout trainability, keyed by the trailing component of the readout's
# ``_target_`` -- the same shape-keyed form as the factor table below.
#
# THIS WAS A DOTTED ``model.readout.trainable`` AND IT WAS ESCAPABLE. Hydra's
# own ``hydra.utils.instantiate`` can be a ``_target_``, so::
#
#     model:
#       _target_: hydra.utils.instantiate
#       _recursive_: false
#       config: { ...the real model, readout frozen... }
#
# constructs exactly the same model with the whole subtree relocated one level
# down. MEASURED: a frozen ``PfaffianReadout`` inside that wrapper VALIDATED,
# while the identical model written directly was refused.
#
# THE DIAGNOSIS IS SHARPER THAN "BAN THE WRAPPER", and the measurement is what
# makes it so. Through the same wrapper, an unadmitted cusp law, a moved
# ``max_order``, an omitted non-finite policy and an unadmitted optimizer were
# ALL still refused -- because those rules identify a thing by WHAT IT IS or by
# key AT ANY DEPTH. The indirection only defeated the one rule that still
# identified a thing by an ANCHORED PATH. So the remedy is to make that rule
# depth-independent rather than to forbid a Hydra feature that is harmless
# against every other rule here.
#
# Note carefully which property does the work: DEPTH-INDEPENDENCE, not
# target-matching. See :func:`_iter_readout_nodes` for the mutant that measured
# this, and for what each of the two nets uniquely catches.
_REQUIRED_READOUT_TRAINABILITY: dict[str, tuple[str, str]] = {
    "PfaffianReadout": (
        "trainable",
        "trainable weighted Pfaffian readout (literal control)",
    ),
}

# The key net's requirement, for a mapping arriving under TPENWaveFunction's own
# ``readout`` keyword regardless of what (or whether) it declares a target.
_READOUT_KEY_REQUIREMENT: tuple[str, str] = (
    "trainable",
    "trainable weighted Pfaffian readout (literal control)",
)

# Factor trainability, keyed by the trailing component of the factor _target_.
_REQUIRED_FACTOR_TRAINABILITY: dict[str, tuple[str, str]] = {
    "ElectronElectronCusp": (
        "trainable_range",
        "both e-e ranges remain trainable in every arm (A10)",
    ),
    "ElectronNucleusCusp": (
        "law.trainable",
        "the e-n curvature law is trainable (literal control, e-n factor)",
    ),
    # Registered when the factor landed rather than when a config first uses
    # it. The failure this table exists to prevent -- a coefficient silently
    # frozen at its initial value, absent from named_parameters() and
    # state_dict() and therefore unlogged -- is available to this factor the
    # moment someone adds it to a config, and a rule added later would arrive
    # after the run that needed it.
    # NOT ACTIVE IN ANY SHIPPED CONFIGURATION as of this commit. The factor
    # exists, is exported from `tpen.nn`, and has this rule -- which together
    # look exactly like a factor in use. It is composed into nothing:
    # `experiments/atomistic/he-importance/configs/train.yaml` does not list it
    # among `model.factors`, and no other config does either. Composing it
    # changes the model that produces the science, so it is a scan-design
    # decision rather than a code chore; tracked as its own item. Said here
    # because a capability present but unwired is the kind of thing a later
    # reader assumes is on.
    "BoundedTwoCoefficientJastrow": (
        "trainable",
        "both Jastrow coefficients train; they start at zero, so an inherited "
        "false leaves the factor as the identity for the whole run with nothing "
        "to show for it",
    ),
}

# ADMITTED electron-nucleus cusp laws, keyed by the trailing component of the
# law's ``_target_``.
#
# An ALLOWLIST, for the same reason the callback set is one: the hazard is a law
# nobody thought to forbid. A denylist would admit every cusp law written after
# it, including the next unconstrained-tail variant.
#
# What membership MEANS here is a physical property, not a preference: the law's
# outer radial slope is negative for every nucleus BY CONSTRUCTION, so no
# training trajectory can reach a growing, non-normalizable tail.
# `CurvatureElectronNucleusCuspLaw` is deliberately ABSENT. It is not
# deprecated and remains correct for `experiments/atomistic/he-v1`, which is
# written in its coordinates -- but its own docstring records that it does not
# enforce ``c/d < Z``, so an HI arm selecting it could cross the sign change
# mid-run with nothing raising. Refusing it at validation converts that into a
# config error before anything is constructed, which is the only point at which
# it is cheap to notice.
_ADMITTED_ELECTRON_NUCLEUS_LAWS: dict[str, str] = {
    "TailSafeElectronNucleusCuspLaw": (
        "coordinates the curvature as c = d (Z - kappa), so the outer slope is "
        "-kappa < 0 for every nucleus at every point in training"
    ),
}

# Absent or null means no clipping, which is what the study requires, so this
# one is checked for absence of a VALUE rather than for the key. The PATH is
# kept for messages -- it is where a reader expects to find the knob -- but the
# check is on the KEY at any depth, because the update method owns and applies
# the clip. See :func:`_sweep_gradient_clip`.
_GLOBAL_CLIP_PATH = "trainer.gradient_clip_norm"
_GLOBAL_CLIP_KEY = "gradient_clip_norm"


def _select(tree: Any, dotted: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for a dotted path in a plain container tree."""

    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _sweep_frozen_architecture(resolved_tree: Any) -> list[Rejection]:
    """Reject a configuration that moves a coordinate no arm may move."""

    if not isinstance(resolved_tree, Mapping):
        return []
    rejections: list[Rejection] = []

    model = resolved_tree.get("model")
    if isinstance(model, (Mapping, list)):
        for path, key, value in iter_nodes({"model": model}):
            if key not in _FROZEN_MODEL_KEYS:
                continue
            expected, authority = _FROZEN_MODEL_KEYS[key]
            if value != expected:
                rejections.append(
                    Rejection(
                        rule="frozen-coordinate",
                        tree="resolved",
                        path=path,
                        detail=f"expected {expected!r}, got {value!r}: {authority}",
                    )
                )

    rejections.extend(_sweep_trainability(resolved_tree))
    return rejections


def _sweep_frozen_scalars(resolved_tree: Any) -> list[Rejection]:
    """Reject a frozen scalar coordinate wherever it is declared or consumed.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per divergent value, at any depth in any section.

    Notes
    -----
    See :data:`_FROZEN_SCALAR_KEYS` for the measurement. Short version: the
    dotted rule checked where the value is DECLARED and the components are built
    from where it is CONSUMED, so `model.embedding.spatial_dim: 2` validated
    beside a compliant `system.spatial_dim: 3`.

    RUN ONCE over the whole tree, NOT per component view, and here that is a
    genuine de-duplication rather than a simplification. A per-view traversal
    would reach `runner.sampler.dtype` from the root view AND from the runner
    view, and both would report it at the IDENTICAL path -- one field, two
    indistinguishable findings. The clip rule differs only because it was never
    inside the view loop.
    """

    rejections: list[Rejection] = []
    # BY KEY AT ANY DEPTH. Absence is still fine -- a node that does not declare
    # the coordinate inherits a correct default -- so only a divergent value is
    # refused, wherever in the tree it is declared or consumed.
    for path, key, value in iter_nodes(resolved_tree):
        spec = _FROZEN_SCALAR_KEYS.get(key) if isinstance(key, str) else None
        if spec is None:
            continue
        expected, authority = spec
        if isinstance(expected, float) or isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = isinstance(value, (int, float)) and float(value) == float(expected)
        else:
            matches = value == expected
        if not matches:
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path=path,
                    detail=f"expected {expected!r}, got {value!r}: {authority}",
                )
            )

    return rejections


def _sweep_gradient_clip(resolved_tree: Any) -> list[Rejection]:
    """Reject a non-null gradient clip wherever it is configured.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per non-null clip, at any depth in any section.

    Notes
    -----
    BY KEY AT ANY DEPTH, not at :data:`_GLOBAL_CLIP_PATH`, and the reason is a
    measured escape rather than tidiness. Clipping is not owned by the trainer:
    ``LegacyAutogradUpdate.__init__`` takes ``gradient_clip_norm`` and is the
    object that APPLIES it, so a ``trainer.update_method`` block naming that
    admitted class with ``gradient_clip_norm: 1.0`` clipped every update while
    ``trainer.gradient_clip_norm`` sat compliantly at null. A single dotted path
    checks the field the study happens to spell today, not the knob.

    NULL AND ABSENT ARE BOTH FINE, which is what every shipped configuration
    does: the control writes ``gradient_clip_norm: null`` explicitly to say so.
    Only a value is refused.

    Run ONCE over the whole tree rather than per component view. That is a
    simplification, NOT a de-duplication, and the difference is worth stating
    because the obvious reading is wrong: the shipped config writes
    ``runner.trainer: ${trainer}``, so the RESOLVED tree genuinely contains the
    field at two paths and one configured clip is reported twice either way.
    The per-view traversal produced the same two. Reporting both is the honest
    behaviour rather than a defect -- when a runner carries a LITERAL copy
    instead of an interpolation, those are two independent fields and naming
    only one would send the reader to the wrong place.
    """

    rejections: list[Rejection] = []
    for path, key, value in iter_nodes(resolved_tree):
        if key != _GLOBAL_CLIP_KEY or value is None:
            continue
        rejections.append(
            Rejection(
                rule="frozen-coordinate",
                tree="resolved",
                path=path,
                detail=(
                    f"global gradient clipping is not part of the study's objective/protection "
                    f"policy, got {value!r}. The update-norm bounds of SR and SPRING are "
                    "parameter-coordinate-dependent protections and are not this knob. "
                    f"Checked by key at any depth, not only at {_GLOBAL_CLIP_PATH!r}: the "
                    "update method owns and applies the clip, so an admitted update rule can "
                    "carry one while the trainer field stays null"
                ),
            )
        )
    return rejections


def _sweep_positional_construction(resolved_tree: Any) -> list[Rejection]:
    """Reject Hydra positional construction anywhere in the configuration.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per ``_args_`` node.

    Notes
    -----
    WHY A FAMILY REFUSAL RATHER THAN A WIDER TRAVERSAL. Every component rule in
    this schema identifies a component by the KEY it hangs from -- ``model``,
    ``trainer``, ``optimizer``, ``model.readout.trainable``,
    ``model.factors[i].law``. ``_args_`` supplies constructor arguments
    POSITIONALLY, and a positional argument has an index rather than a name.
    ``tpen.runner.Train`` takes ``(model, sampler, hamiltonian_terms, optimizer,
    trainer)``, so ``runner._args_`` builds exactly those five components with
    no key for any rule to match. MEASURED: all five of the divergences the
    component views refuse by keyword -- a frozen readout, a global clip, an
    omitted non-finite policy, an unadmitted cusp law, an unadmitted optimizer
    -- were ACCEPTED when the same content was passed positionally.

    This is the KEY-VERSUS-VALUE failure that produced the target-allowlist gap,
    one level up: a rule that reads names is defeated by a caller who supplies
    the same thing without one.

    The alternative is to make every rule shape-based -- identify a component by
    its ``_target_`` rather than by its key. That is a larger and riskier change
    than this slice is scoped for, and it would widen the blast radius of five
    rules at once. Refusing the family and requiring a schema change to re-admit
    it is the same doctrine :data:`STOP_RULE_SURFACE` applies to a surface whose
    hazard cannot be inspected at schema time.

    OVER-RESTRICTION MEASURED, not assumed: ``_args_`` appears in NO
    configuration under ``experiments/`` and in no test fixture, so this refuses
    nothing that exists. If a future arm genuinely needs positional
    construction, the refusal is loud, names this function, and the remedy is to
    make the rules shape-based rather than to delete this one.

    WHY THE FAMILY REFUSAL IS THE RIGHT INSTRUMENT, audited independently
    rather than argued from taste. Hydra keyword construction binds a config key
    to a CONSUMER PARAMETER NAME, and every consumer on the constructed path has
    a strict signature: ``Train(model, sampler, hamiltonian_terms, optimizer,
    trainer, load=None)``; ``VMCTrainer`` with exact names and no aliases;
    ``LegacyAutogradUpdate(gradient_clip_norm)``; ``TPENWaveFunction``, whose
    ``**kwargs`` forwards only to a strict ``EquivariantMap`` so an unknown key
    is a TypeError rather than being stored and swallowed. So for keyword
    supply, **the config key IS the parameter name at the consumer** -- which is
    what grounds every key-at-any-depth rule in this module in construction
    semantics rather than in convention.

    ``_args_`` was the only config-reachable way to bind a parameter with NO
    key. The alternate positional routes all dead-end: a ``functools.partial``
    target needs ``_args_`` for its own positional-only argument, and
    ``getattr``/``attrgetter``/``methodcaller`` are positional-only too. A
    resolution-time ``_args_``, materialised through a resolver, is refused as
    well, because the resolved sweep walks materialised containers.

    RESIDUAL, CORRECTED. An earlier version of this paragraph named a model hung
    from ``runner.net``. **That was the comfortable member and it is not
    reachable at all**: ``Train`` has no such parameter, so it is a TypeError at
    construction. Naming it read as candour while leaving the reachable
    residuals unstated -- the second time in this slice that happened, and the
    reviewer caught both.

    The reachable residuals of this same class, as of this commit, are:

    - **Generic importers carrying a module path as DATA. NARROWED, NOT
      CLOSED, and the earlier text here said "Closed" -- which was wrong.**
      Widening the module-identity check to every string value (see
      :func:`_sweep_target_values`) catches the DOTTED spelling wherever it
      appears. It does not catch a path that is never spelled dotted. MEASURED
      at head ``745de1e``, all three validating end-to-end:
      ``import_module(name=".hi_manifest", package="tpen")`` splits the path
      across two strings; ``runpy.run_path(path_name="tpen/hi_manifest.py")``
      uses the FILESYSTEM spelling and executes the module source outright;
      ``builtins.__import__(name="hi_manifest", globals={"__package__": "tpen"},
      level=1)`` does the same through the import hook.

      The false step was the doctrine sentence, not the code: "the module path
      has to appear SOMEWHERE as a string" is true only if it appears WHOLE.
      Split across two arguments, or written as a file path, it does not.

      Filed as its own item rather than patched here: the filesystem class is
      not closable by string identity at all -- absolute paths, ``./`` prefixes,
      symlinks and case-insensitive filesystems all spell the same file -- so
      the remedy has to govern the SLOT or the CONSUMER rather than enumerate
      spellings, and that is a new production surface.

      Impact: the module is imported or executed on the training path, which is
      the hazard the rule names.

      **THE OLD BOUND HERE WAS FALSE AND IS RETRACTED.** It said the reference
      NUMBERS were safe because ``_args_`` is refused so "no config-only shape
      can CALL the loader". That bounded a hazard by ONE LOADER'S NAME -- the
      same failure this whole slice is about, committed in an impact bound
      instead of in a rule, and it shipped. MEASURED by an independent verifier
      at ``96f64f6``::

          runner:
            load:
              _target_: omegaconf.OmegaConf.load
              file_: experiments/atomistic/he-importance/manifests/evaluation.yaml

      That VALIDATES, and instantiating it returns ``reference.energy ==
      -2.9037243770341195``. It needs no module path and trips no token: the
      manifest's own file path tokenizes to nothing this schema denies. The
      numbers are reachable without touching ``tpen.hi_manifest`` at all.

      Filed as its own item under Amendment C rather than fixed here. It does
      not falsify the acceptance contract, which explicitly excludes runtime
      isolation of the reference -- but nothing in this module may claim the
      numbers are out of reach, and until that item lands, they are not.
    - **Resolvers supplying a denied capability under an undenied name.** Closed
      by moving to an allowlist; see ``allowed_resolvers`` on
      :data:`HI_TRAIN_POLICY`.
    - **Still open: a component reached through a keyword no rule names, in a
      slot whose consumer does NOT have a strict signature.** No such slot is
      known on the constructed path today, and the strict-signature audit above
      is what makes that a bounded claim rather than an assumption. It is an
      assumption about every consumer ADDED LATER, and nothing enforces it.
    """

    rejections: list[Rejection] = []
    for path, key, _value in iter_nodes(resolved_tree):
        if key != "_args_":
            continue
        rejections.append(
            Rejection(
                rule="positional-construction",
                tree="resolved",
                path=path,
                detail=(
                    "Hydra positional construction is not permitted in this schema. Every "
                    "component rule identifies what it judges by the KEY the component hangs "
                    "from, and a positional argument has an index instead of a name -- so "
                    "`_args_` builds the same components with nothing for the rules to match. "
                    "Pass every component as a keyword. Re-admitting `_args_` requires making "
                    "those rules shape-based, which is a schema change rather than a config one"
                ),
            )
        )
    return rejections


def _iter_factor_nodes(resolved_tree: Any) -> list[tuple[str, Mapping[str, Any], str]]:
    """Find every configured factor under ``model``, BY SHAPE rather than by path.

    Parameters
    ----------
    resolved_tree : Any
        The tree to search -- the configuration root, or a component view.

    Returns
    -------
    list of (str, Mapping, str)
        ``(path, node, class_name)`` for each mapping under ``model`` whose
        ``_target_`` trailing component names a factor this schema has a rule
        for. ``class_name`` is that trailing component.

    Notes
    -----
    THE PATH-SHAPED VERSION WAS ESCAPABLE WITH ORDINARY KEYWORD CONFIG, no
    ``_args_`` involved. Both factor rules read ``model.factors`` and returned
    immediately unless it was a LIST. ``TPENWaveFunction`` accepts any iterable
    and normalizes it, so::

        model.factors:
          _target_: torch.nn.ModuleList
          modules: [ ...the same factors... ]

    is a valid, constructible configuration in which ``model.factors`` is a
    MAPPING. MEASURED: an unadmitted ``CurvatureElectronNucleusCuspLaw`` and a
    frozen electron-electron factor both validated inside that wrapper.

    Refusing the wrapper would be the third instance of the same mistake this
    slice keeps making -- naming the shapes I happened to think of. A factor is
    identified by WHAT IT IS, its ``_target_``, not by the container it arrives
    in, so any container works and none has to be enumerated.

    Scoped to the ``model`` subtree rather than the whole tree, because that is
    where a factor is a factor. The same class appearing in, say, a diagnostic's
    arguments is not the model's factor list and has no trainability contract.
    """

    model_found, model = _select(resolved_tree, "model")
    if not model_found or not isinstance(model, (Mapping, list)):
        return []
    found: list[tuple[str, Mapping[str, Any], str]] = []
    for path, _key, value in iter_nodes({"model": model}):
        if not isinstance(value, Mapping):
            continue
        target = value.get("_target_")
        if not isinstance(target, str):
            continue
        found.append((path, value, target.rsplit(".", 1)[-1]))
    return found


def _sweep_hamiltonian_terms(resolved_tree: Any) -> list[Rejection]:
    """Reject a Hamiltonian term that moves a coordinate no arm may move.

    Parameters
    ----------
    resolved_tree : Any
        The tree to check -- the configuration root, or a component view.

    Returns
    -------
    list of Rejection
        One rejection per divergent coordinate, at any depth under
        ``hamiltonian_terms``, in whatever container the terms arrive in.

    Notes
    -----
    See :data:`_FROZEN_TERM_COORDINATES` for the measurement that motivated
    matching on the term's ``_target_`` rather than on a dotted path. The short
    version: the terms may be a Mapping or a Sequence, and when they are a
    Mapping the KEYS are the config author's choice. A rule keyed on
    ``electron_nucleus`` checks a name the author was never obliged to use.
    """

    found, terms = _select(resolved_tree, "hamiltonian_terms")
    if not found or not isinstance(terms, (Mapping, list)):
        return []
    rejections: list[Rejection] = []
    for path, _key, value in iter_nodes({"hamiltonian_terms": terms}):
        if not isinstance(value, Mapping):
            continue
        target = value.get("_target_")
        if not isinstance(target, str):
            continue
        coordinates = _FROZEN_TERM_COORDINATES.get(target.rsplit(".", 1)[-1])
        if coordinates is None:
            continue
        for argument, (expected, authority) in coordinates.items():
            if argument not in value:
                # Absent means the constructor's own default applies, and for
                # every coordinate in this table that default IS the study's
                # value. Requiring the declaration would refuse configs that
                # simply omit it.
                continue
            actual = value[argument]
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and float(actual) == float(expected)
            )
            if not matches:
                rejections.append(
                    Rejection(
                        rule="frozen-coordinate",
                        tree="resolved",
                        path=f"{path}.{argument}",
                        detail=f"expected {expected!r}, got {actual!r}: {authority}",
                    )
                )
    return rejections


def _iter_readout_nodes(
    resolved_tree: Any,
) -> list[tuple[str, Mapping[str, Any], str, str]]:
    """Find every configured readout under ``model``, by TARGET *and* by key.

    Returns
    -------
    list of (str, Mapping, str, str)
        ``(path, node, required_argument, authority)``, de-duplicated by path.

    Notes
    -----
    TWO NETS ON PURPOSE, and this is the one place in this module where a
    key-shaped rule is KEPT rather than replaced.

    WHAT ACTUALLY CLOSED THE ESCAPE, stated precisely because the obvious
    reading is wrong. Hydra's own ``hydra.utils.instantiate`` can be a
    ``_target_``, so wrapping the model in it relocates the whole subtree one
    level down and the old DOTTED ``model.readout.trainable`` named nothing.
    MEASURED: a frozen ``PfaffianReadout`` inside that wrapper validated and
    constructed a readout with ZERO parameters, forwarding finitely -- a run
    that trains its whole budget with a permanently frozen readout.

    That escape is closed by the KEY net walking the ``model`` subtree at any
    depth, NOT by the shape net: the wrapped readout still arrives under the key
    ``readout``, at ``model.config.readout``. **Removing the shape net leaves
    the wrapped case refused.** Measured, after a mutant removing it changed no
    behaviour on that probe -- the mutant was sound and the attribution was not.

    So each net is kept for a case only IT catches, and neither is decoration:

    - KEY net only: a readout written with NO ``_target_``. Unlike a Hamiltonian
      term, which ``_validate_hamiltonian_term`` refuses loudly for lacking a
      callable ``local_energy``, a bare readout mapping has no comparably crisp
      construction-time backstop.
    - SHAPE net only: a ``PfaffianReadout`` under a key that is not ``readout``,
      e.g. ``model.head``. Recorded honestly as a PRECONSTRUCTION gap rather
      than a live escape: ``TPENWaveFunction`` takes ``**kwargs`` so the key is
      swallowed, and ``readout`` is a required keyword, so MOVING the readout
      there fails at construction. It is closed for the same reason F2's SR case
      was -- "another component happens to refuse it" is a property of today's
      constructors, not a rule.

    Union, not replacement. Every other rule in this module moved from name to
    shape; this one is depth-independent in BOTH, and the asymmetry is
    deliberate.
    """

    model_found, model = _select(resolved_tree, "model")
    if not model_found or not isinstance(model, (Mapping, list)):
        return []
    seen: dict[str, tuple[str, Mapping[str, Any], str, str]] = {}
    for path, key, value in iter_nodes({"model": model}):
        if not isinstance(value, Mapping):
            continue
        target = value.get("_target_")
        spec = None
        if isinstance(target, str):
            spec = _REQUIRED_READOUT_TRAINABILITY.get(target.rsplit(".", 1)[-1])
        if spec is None and key == "readout":
            # The key net. ``readout`` is TPENWaveFunction's own keyword, so a
            # mapping arriving under it is the readout whatever it declares.
            spec = _READOUT_KEY_REQUIREMENT
        if spec is None:
            continue
        relative, authority = spec
        seen.setdefault(path, (path, value, relative, authority))
    return list(seen.values())


def _sweep_trainability(resolved_tree: Mapping[str, Any]) -> list[Rejection]:
    """Require every trainability flag to be declared true, never inherited."""

    rejections: list[Rejection] = []

    # Scoped to configurations that actually declare the component. A config
    # with no readout at all is INCOMPLETE, which is a different defect from a
    # readout whose trainability was silently inherited -- and it is the latter
    # this rule exists to catch. Conflating them would make every partial config
    # report a trainability violation it does not have. Shape-based discovery
    # gives that scoping for free: a config with no readout has no node to match.
    for path, node, relative, authority in _iter_readout_nodes(resolved_tree):
        found, value = _select(node, relative)
        if not found:
            rejections.append(
                Rejection(
                    rule="undeclared-trainability",
                    tree="resolved",
                    path=f"{path}.{relative}",
                    detail=(
                        f"must be declared explicitly ({authority}); the class default is "
                        "the OPPOSITE, and an inherited false is invisible -- the "
                        "parameter appears in neither named_parameters() nor state_dict(), so "
                        "nothing logs it and no gradient touches it"
                    ),
                )
            )
        elif value is not True:
            rejections.append(
                Rejection(
                    rule="undeclared-trainability",
                    tree="resolved",
                    path=f"{path}.{relative}",
                    detail=f"expected true, got {value!r}: {authority}",
                )
            )

    for path, factor, class_name in _iter_factor_nodes(resolved_tree):
        spec = _REQUIRED_FACTOR_TRAINABILITY.get(class_name)
        if spec is None:
            continue
        relative, authority = spec
        found, value = _select(factor, relative)
        if not found or value is not True:
            rejections.append(
                Rejection(
                    rule="undeclared-trainability",
                    tree="resolved",
                    path=f"{path}.{relative}",
                    detail=(
                        f"must be declared explicitly and true ({authority}); "
                        f"{'absent' if not found else repr(value)}"
                    ),
                )
            )
    return rejections


# ---------------------------------------------------------------------------
# Rank-invariant preconstruction
# ---------------------------------------------------------------------------
# Admissible non-finite local-energy policies, and the one this study defaults
# to nothing. DECLARATION IS REQUIRED: there is no inherited value, because the
# inherited value would be the historical "mask" and masking is a
# known-biased estimator rather than a neutral fallback. Non-finite local
# energies occur where the local energy is pathological -- near nodes, at
# coalescence, in the tail -- so dropping them selects a subsample
# systematically and biases the energy by an uncharacterised amount. A count of
# how many rows were dropped does not recover that bias.
#
# So "mask" stays REACHABLE, because a scientist may knowingly want it, but only
# by writing it down where this schema can see it.
_ADMITTED_NONFINITE_POLICIES: frozenset[str] = frozenset({"fail", "mask"})
_NONFINITE_POLICY_PATH = "trainer.nonfinite_local_energy_policy"
_NONFINITE_POLICY_KEY = "nonfinite_local_energy_policy"


def _sweep_nonfinite_local_energy_policy(resolved_tree: Any) -> list[Rejection]:
    """Require an explicit, admissible non-finite local-energy policy.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection if the policy is absent or not admissible.
    """

    # Scoped to configurations that actually declare a trainer, matching the
    # trainability rule's precedent that "a config with no readout is not a
    # trainability violation". A configuration with no `trainer` section is not
    # configuring training at all, so there is no estimator for it to declare
    # and demanding one would refuse valid partial configs -- an
    # over-restriction that would surface as a run that cannot start.
    trainer_found, trainer_section = _select(resolved_tree, "trainer")
    if not trainer_found or not isinstance(trainer_section, Mapping):
        return []

    # Every declared policy is checked for admissibility BY KEY AT ANY DEPTH,
    # for the same reason the update-method rule is: a factory wrapper relocates
    # the real trainer one level down, and a rule anchored to
    # ``trainer.nonfinite_local_energy_policy`` would read the wrapper's copy
    # while the constructed trainer used the inner one.
    rejections: list[Rejection] = []
    declared_anywhere = False
    for policy_path, key, value in iter_nodes(resolved_tree):
        if key != _NONFINITE_POLICY_KEY or value is None:
            continue
        declared_anywhere = True
        if value not in _ADMITTED_NONFINITE_POLICIES:
            rejections.append(
                Rejection(
                    rule="undeclared-nonfinite-policy",
                    tree="resolved",
                    path=policy_path,
                    detail=(
                        f"{value!r} is not admissible; expected one of "
                        f"{sorted(_ADMITTED_NONFINITE_POLICIES)}"
                    ),
                )
            )
    if rejections:
        return rejections
    if not declared_anywhere:
        return [
            Rejection(
                rule="undeclared-nonfinite-policy",
                tree="resolved",
                path=_NONFINITE_POLICY_PATH,
                detail=(
                    "must be declared explicitly; admissible values are "
                    f"{sorted(_ADMITTED_NONFINITE_POLICIES)}. There is no default here on "
                    "purpose: the inherited behaviour is to MASK non-finite local-energy "
                    "rows, which drops a systematically selected subsample and biases the "
                    "energy estimator by an uncharacterised amount. A run must say which "
                    "estimator it is using"
                ),
            )
        ]
    return []


def _sweep_electron_nucleus_law(resolved_tree: Any) -> list[Rejection]:
    """Reject an electron-nucleus cusp law outside the admitted set.

    Separate from `_sweep_trainability`, which asks whether the law is TRAINED.
    This asks whether it is the RIGHT LAW, and the two are independent: a
    trainable unconstrained-tail law satisfies that rule completely while still
    being able to train its way into a non-normalizable tail.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per factor naming an unadmitted or undeclared law.
    """

    rejections: list[Rejection] = []
    for factor_path, factor, class_name in _iter_factor_nodes(resolved_tree):
        if class_name != "ElectronNucleusCusp":
            continue
        found, law = _select(factor, "law")
        if not found or not isinstance(law, Mapping):
            # An absent law is already fatal through the trainability rule,
            # which requires `law.trainable` to be declared true. Not repeated
            # here: two rejections for one omission would read as two defects.
            continue
        law_target = law.get("_target_")
        path = f"{factor_path}.law._target_"
        if not isinstance(law_target, str):
            rejections.append(
                Rejection(
                    rule="unadmitted-cusp-law",
                    tree="resolved",
                    path=path,
                    detail=(
                        "the electron-nucleus cusp law must declare a _target_; admitted "
                        f"laws are {sorted(_ADMITTED_ELECTRON_NUCLEUS_LAWS)}"
                    ),
                )
            )
            continue
        name = law_target.rsplit(".", 1)[-1]
        if name not in _ADMITTED_ELECTRON_NUCLEUS_LAWS:
            admitted = ", ".join(
                f"{key} ({reason})" for key, reason in sorted(_ADMITTED_ELECTRON_NUCLEUS_LAWS.items())
            )
            rejections.append(
                Rejection(
                    rule="unadmitted-cusp-law",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"{law_target!r} is not admitted for this study. An admitted law must "
                        "guarantee a decaying outer tail for every nucleus by construction, "
                        "rather than leaving the bound to the caller. Admitted: "
                        f"{admitted}"
                    ),
                )
            )
    return rejections


def _sweep_update_method(resolved_tree: Any) -> list[Rejection]:
    """Reject a ``trainer.update_method`` outside the admitted set.

    Parameters
    ----------
    resolved_tree : Any
        The tree to check -- the configuration root, or a component view.

    Returns
    -------
    list of Rejection
        One rejection when the declared update method is not admitted.

    Notes
    -----
    :func:`_sweep_method` qualifies ``optimizer._target_``. This qualifies the
    other half: the rule that performs the update. They are independent
    surfaces, and a configuration naming Adam in one and an unadmitted update
    rule in the other passed the roster completely.

    PRESENCE-SCOPED, like every other component rule. An absent or null
    ``update_method`` resolves to ``LegacyAutogradUpdate`` -- the plain
    optimizer step, which is the admitted method -- and is what every shipped
    configuration does, so requiring the declaration would refuse them all.

    BY KEY AT ANY DEPTH, not at ``trainer.update_method``. The anchored version
    was escapable through a Hydra factory wrapper: with the trainer written as
    ``{_target_: hydra.utils.instantiate, config: <the real trainer>}`` the rule
    found no ``update_method`` on the wrapper and returned, while the real spec
    sat one level down. MEASURED: an unadmitted
    ``StochasticReconfigurationUpdate`` VALIDATED that way.

    It first appeared to be refused, which is the trap worth recording: the
    wrapper also hid ``nonfinite_local_energy_policy``, so a DIFFERENT rule
    fired and the configuration was rejected for an unrelated reason. Restating
    the policy on the wrapper removed that accident and the escape was live.
    **A refusal is only evidence for the rule that produced it** -- assert the
    rule name, not that something was refused.
    """

    admitted = sorted(ADMITTED_UPDATE_METHOD_TARGETS)
    rejections: list[Rejection] = []
    for spec_path, key, spec in iter_nodes(resolved_tree):
        if key != "update_method" or spec is None:
            continue
        rejections.extend(_qualify_update_method(spec_path, spec, admitted))
    return rejections


def _qualify_update_method(
    path: str, spec: Any, admitted: list[str]
) -> list[Rejection]:
    """Qualify one declared update-method spec against the admitted set."""

    if not isinstance(spec, Mapping):
        return [
            Rejection(
                rule="unadmitted-update-method",
                tree="resolved",
                path=path,
                detail=(
                    f"update_method must be a config block declaring a _target_, got "
                    f"{type(spec).__name__}. Admitted: {admitted}. "
                    f"Roster: {_roster_summary()}"
                ),
            )
        ]

    target = spec.get("_target_")
    if not isinstance(target, str):
        return [
            Rejection(
                rule="unadmitted-update-method",
                tree="resolved",
                path=f"{path}._target_",
                detail=(
                    "a declared update_method must name a _target_; there is nothing to "
                    f"qualify otherwise. Admitted: {admitted}. Roster: {_roster_summary()}"
                ),
            )
        ]

    if target not in ADMITTED_UPDATE_METHOD_TARGETS:
        return [
            Rejection(
                rule="unadmitted-update-method",
                tree="resolved",
                path=f"{path}._target_",
                detail=(
                    f"update rule {target!r} is not admitted. The optimizer roster qualifies "
                    "optimizer._target_ only, so an unadmitted update rule reached "
                    "construction with an admitted optimizer beside it. Refused rather than "
                    "replaced, for the same reason an unavailable optimizer is: a method that "
                    "quietly became the default would let a run report that it exercised "
                    f"something it never ran. Admitted: {admitted}. Roster: {_roster_summary()}"
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Closing over the components that are actually CONSTRUCTED
# ---------------------------------------------------------------------------
# THE HOLE THIS CLOSES, and it is the schema's own claim failing rather than a
# new kind of problem. Every component rule above reads a path from the
# configuration ROOT -- `model`, `trainer.gradient_clip_norm`,
# `model.factors[i].law`, `optimizer`. What `tpen.run._instantiate_runner`
# hands to Hydra is `cfg.runner`, and `instantiate` is RECURSIVE, so what gets
# built is whatever hangs under `runner`. In the shipped control config those
# are interpolations (`model: ${model}`) and the two agree -- but nothing
# required that. A runner section carrying its own literal copies was accepted
# with a frozen readout, a global gradient clip, an unadmitted cusp law and an
# OMITTED non-finite policy, while every root field stayed compliant.
#
# WHY VIEWS RATHER THAN ROOT-EQUALITY. Requiring `runner.model == model` would
# refuse every divergence, including ones nobody cares about, and would forbid
# a runner from ever carrying a component the root does not also spell. That
# over-restriction is invisible until a legitimate run cannot start. Re-running
# the component rules over what is constructed refuses exactly the divergences
# the rules already name, and nothing else -- and it needs no new rule, which
# is why the blast radius is the existing rules' blast radius.
#
# COMPONENT KEYS, not a fixed `runner.*` path list: the view is any mapping
# under `runner` that carries at least one component the rules know how to
# judge. In the shipped config the only such node is `runner` itself, so this
# costs one extra pass; a config that nested a component deeper is covered
# without a second rule.
_COMPONENT_KEYS = frozenset(
    {"model", "trainer", "optimizer", "sampler", "hamiltonian_terms"}
)

# The rules that judge a COMPONENT rather than the run as a whole. Deliberately
# omitted: `_sweep_rank_divergent_fields`, which is about `run.run_id` and is a
# property of the run's identity rather than of anything constructed;
# `_sweep_callbacks`, because `_instantiate_runner` already refuses a runner
# that owns callbacks at all; and `_sweep_target_values`, which already walks
# the whole resolved tree and therefore covers every view for free.
_COMPONENT_SWEEPS = (
    _sweep_frozen_architecture,
    _sweep_hamiltonian_terms,
    _sweep_electron_nucleus_law,
    _sweep_nonfinite_local_energy_policy,
    _sweep_update_method,
)


def _reprefix(rejections: list[Rejection], prefix: str) -> list[Rejection]:
    """Re-root each rejection's path so it names where the component was found.

    A finding reported at ``trainer.gradient_clip_norm`` when the offending
    value is at ``runner.trainer.gradient_clip_norm`` sends the reader to a
    field that is compliant, which is worse than no path at all.
    """

    return [replace(rejection, path=f"{prefix}{rejection.path}") for rejection in rejections]


def _sweep_constructed_components(resolved_tree: Any) -> list[Rejection]:
    """Apply the component rules to the root AND to everything under ``runner``.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        Findings from every component view, each path re-rooted at the view.

    Notes
    -----
    The root view keeps its existing semantics exactly, including
    ``require_optimizer=True``: a training configuration that declares no
    method is incomplete. A component view is PRESENCE-SCOPED throughout --
    every rule it runs already skips a component the view does not declare --
    so an evaluation runner carrying no trainer and no optimizer is untouched.

    RESIDUAL, stated rather than papered over: this closes the components the
    rules know how to judge. A component the rules have no rule for is still
    unjudged wherever it appears, at the root as much as under ``runner``, and
    that is a gap in the RULE SET rather than in this traversal.
    """

    rejections: list[Rejection] = []
    for component_sweep in _COMPONENT_SWEEPS:
        rejections.extend(component_sweep(resolved_tree))
    rejections.extend(_sweep_method(resolved_tree))

    if not isinstance(resolved_tree, Mapping):
        return rejections
    runner = resolved_tree.get("runner")
    if not isinstance(runner, Mapping):
        return rejections

    for path, _key, value in iter_nodes({"runner": runner}):
        if not isinstance(value, Mapping) or not (_COMPONENT_KEYS & set(value)):
            continue
        for component_sweep in _COMPONENT_SWEEPS:
            rejections.extend(_reprefix(component_sweep(value), f"{path}."))
        rejections.extend(
            _reprefix(_sweep_method(value, require_optimizer=False), f"{path}.")
        )
    return rejections


def _sweep_rank_divergent_fields(resolved_tree: Any) -> list[Rejection]:
    """Reject fields that would resolve to a different value in each process.

    Notes
    -----
    MEASURED in ``tpen/artifacts.py``: ``generate_run_id`` returns
    ``f"{timestamp}_{slug}_{uuid4().hex[:6]}"``, and ``prepare_run_context``
    calls it whenever ``run.run_id`` resolves to ``None``. The suffix is
    RANDOM, so a null ``run_id`` does not merely risk divergence under clock
    skew -- it produces a different identifier in every process, always. Under
    a distributed launch each rank would then write to its own run directory
    and the run would have no single identity.

    This is the acceptance contract's second falsifier, "rank inputs differ",
    and it is a property of the existing code rather than of anything this
    schema adds. The remedy is for the materializer to assign the run id, which
    is L2's job; the schema's job is to refuse a config that leaves it to
    chance.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    run = resolved_tree.get("run")
    if not isinstance(run, Mapping):
        return []
    if run.get("run_id") is not None:
        return []
    return [
        Rejection(
            rule="rank-divergent-field",
            tree="resolved",
            path="run.run_id",
            detail=(
                "run.run_id must be assigned explicitly. A null run_id is filled in by "
                "generate_run_id, whose value ends in uuid4().hex[:6] -- so every process "
                "computes a DIFFERENT identifier and each rank would write to its own run "
                "directory. The resolved configuration must be identical on all ranks"
            ),
        )
    ]


def canonical_train_identity(cfg: DictConfig) -> str:
    """Return the digest every rank must agree on for one training run.

    Parameters
    ----------
    cfg : DictConfig
        A resolved training configuration.

    Returns
    -------
    str
        Hex SHA-256 of the canonical rendering of the resolved configuration.

    Notes
    -----
    Two ranks launched from the same file and overrides must produce the same
    value. What makes that true is not this function but the two checks beside
    it: forbidden resolvers keep resolution a pure function of the file, and
    the rank-divergent-field rule keeps ``run.run_id`` from being drawn per
    process. Without those, this digest would faithfully report a difference
    and say nothing about its cause.

    The digest covers the whole resolved configuration deliberately. Excluding
    fields to make ranks agree would be arranging for the answer.
    """

    return canonical_digest(OmegaConf.to_container(cfg, resolve=True))


def _roster_summary() -> str:
    """Render the roster so a refusal states every method's admission status."""

    return "; ".join(
        f"{entry.method}={'admitted' if entry.admitted else 'unavailable'} ({entry.requires})"
        for entry in HI_METHOD_ROSTER
    )


def _sweep_method(resolved_tree: Any, *, require_optimizer: bool = True) -> list[Rejection]:
    """Reject a training configuration whose optimizer method is not admitted.

    Parameters
    ----------
    resolved_tree : Any
        The tree to check, which is either the configuration root or one of the
        component views under ``runner``.
    require_optimizer : bool, optional
        Whether an ABSENT optimizer is itself a violation. True at the root,
        where a training configuration that declares no method is incomplete.
        False on a component view, where absence means the view simply does not
        carry an optimizer -- demanding one there would refuse every runner that
        is not a trainer, which is an over-restriction that surfaces as a run
        that cannot start rather than as a red test.

    Notes
    -----
    An unadmitted method is refused rather than replaced. The plan of record
    requires an unavailable cell to stay visibly unavailable; substituting Adam
    would let a run report that it exercised a method it never ran, which is a
    worse failure than not running at all.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    optimizer = resolved_tree.get("optimizer")
    if optimizer is None:
        if not require_optimizer:
            return []
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer",
                detail=(
                    "a training configuration must declare its optimizer method. "
                    f"Roster: {_roster_summary()}"
                ),
            )
        ]
    if not isinstance(optimizer, Mapping):
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer",
                detail=f"optimizer must be a config block, got {type(optimizer).__name__}",
            )
        ]

    target = optimizer.get("_target_")
    if not isinstance(target, str):
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer._target_",
                detail=f"optimizer must declare a _target_. Roster: {_roster_summary()}",
            )
        ]

    if target not in ADMITTED_METHOD_TARGETS:
        return [
            Rejection(
                rule="unadmitted-method",
                tree="resolved",
                path="optimizer._target_",
                detail=(
                    f"method {target!r} is not admitted. It is refused rather than replaced: "
                    "an unavailable method must stay visibly unavailable, never silently "
                    f"become Adam. Roster: {_roster_summary()}"
                ),
            )
        ]

    return _sweep_adam_coordinates(optimizer)


def _sweep_adam_coordinates(optimizer: Mapping[str, Any]) -> list[Rejection]:
    """Reject an Adam block that varies a coordinate no arm may vary."""

    rejections: list[Rejection] = []
    for key, expected in _FIXED_ADAM_COORDINATES.items():
        if key not in optimizer:
            # Absent means the library default applies, and the fixed values
            # here ARE the library defaults. Requiring them to be spelled out
            # would reject every config that simply omits them.
            continue
        actual = optimizer[key]
        if float(actual) != float(expected):
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path=f"optimizer.{key}",
                    detail=(
                        f"Adam {key} is fixed at {expected} for every cell in the optimizer "
                        f"grid, got {actual!r}. Only lr and beta2 are scanned"
                    ),
                )
            )

    betas = optimizer.get("betas")
    if isinstance(betas, Sequence) and not isinstance(betas, (str, bytes)) and betas:
        if float(betas[0]) != _FIXED_ADAM_BETA1:
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path="optimizer.betas[0]",
                    detail=(
                        f"Adam beta1 is fixed at {_FIXED_ADAM_BETA1} for every cell, got "
                        f"{betas[0]!r}. beta2 is the scanned moment, beta1 is not"
                    ),
                )
            )
    return rejections


def validate_hi_train_config(cfg: DictConfig, *, env: Mapping[str, str] | None = None) -> None:
    """Refuse a helium-importance training configuration that violates its schema.

    Parameters
    ----------
    cfg : DictConfig
        The loaded training configuration, before any construction.
    env : Mapping of str to str, optional
        The launch environment to audit. Defaults to ``os.environ``.

    Raises
    ------
    ClosedSchemaError
        When the configuration declares :data:`HI_TRAIN_SCHEMA` and violates it.
        Every finding is reported, not only the first.

    Notes
    -----
    A configuration that declares no schema, or a different one, is returned
    unvalidated -- see the module docstring on why the firewall is opt-in.

    KNOWN AND DEFERRED, stated here because this is where a reader meets the
    "refuse before anything is constructed" claim and that claim is narrower
    than it sounds. **Resolution is not a read.** A registered OmegaConf
    resolver RUNS during ``to_container(resolve=True)`` below, and
    ``tpen.config`` registers ``tpen.basis_feature_dim``, which calls
    ``hydra.utils.instantiate`` on its argument. MEASURED at head ``38edc53``:
    a configuration pointing that resolver at ``builtins.open`` was correctly
    REFUSED -- and the file it named had already been created, because the
    callable ran while the tree was being resolved for the sweeps that would
    refuse it.

    So the claim holds for CONSTRUCTION and not for RESOLUTION. The closed
    resolver allowlist means every such configuration IS refused; what is not
    guaranteed is that it is refused BEFORE the resolver runs.

    Deliberately NOT fixed in this slice. It is a new production surface rather
    than one of the four construction-path holes this slice was scoped to, so
    it is FILED rather than absorbed -- the remedy is a handful of lines and is
    available, but scope is the manager's call and not this lane's to widen.
    Found by an independent reviewer and labelled independent.

    The launch environment is audited alongside the configuration because the
    reference-energy firewall names it as one of the surfaces a reference must
    not enter, next to training configs, runner inputs, checkpoint metadata and
    decision logic. A config-only check would leave one of the five open.
    """

    declared = declared_schema(cfg)
    if declared != HI_TRAIN_SCHEMA:
        if is_hi_family(cfg):
            # The config says it is helium-importance but did not declare the
            # schema. Refusing loudly here is the whole point: silently
            # returning would give a real HI run zero enforcement.
            raise ClosedSchemaError(
                [
                    Rejection(
                        rule="undeclared-schema",
                        tree="raw",
                        path=SCHEMA_KEY,
                        detail=(
                            f"experiment.name is {HI_EXPERIMENT_NAME!r}, so this is a "
                            f"helium-importance configuration and must declare "
                            f"{SCHEMA_KEY}: {HI_TRAIN_SCHEMA}. Declared: {declared!r}. "
                            "The marker is not optional for this family -- a closed schema "
                            "that applies only when a config remembers to ask for it is not "
                            "closed, and the omission would otherwise be silent"
                        ),
                    )
                ]
            )
        return

    environment = os.environ if env is None else env
    raw_tree = OmegaConf.to_container(cfg, resolve=False)

    # The raw sweep runs FIRST and unconditionally. Resolution can fail, and if
    # the raw findings were collected after it, a single broken interpolation
    # would suppress every forbidden reference the raw tree already showed --
    # the finding and the thing that hides it would share a failure domain. A
    # config that both fails to resolve and carries a reference must report the
    # reference, because that is the finding that stops a run from happening.
    rejections = list(sweep_raw(raw_tree, HI_TRAIN_POLICY))
    rejections.extend(sweep_environment(environment, HI_TRAIN_POLICY))

    # Resolution failure is itself a rejection, which keeps every
    # preconstruction failure a single exception type for the caller.
    try:
        resolved_tree = OmegaConf.to_container(cfg, resolve=True)
    except Exception as error:  # noqa: BLE001 - OmegaConf raises several unrelated types
        rejections.append(
            Rejection(
                rule="unresolvable",
                tree="resolved",
                path="<root>",
                detail=f"configuration does not resolve: {type(error).__name__}: {error}",
            )
        )
        raise ClosedSchemaError(rejections) from error

    rejections.extend(sweep_resolved(resolved_tree, HI_TRAIN_POLICY))
    rejections.extend(_sweep_callbacks(resolved_tree))
    rejections.extend(_sweep_target_values(resolved_tree))
    rejections.extend(_sweep_constructed_components(resolved_tree))
    rejections.extend(_sweep_frozen_scalars(resolved_tree))
    rejections.extend(_sweep_gradient_clip(resolved_tree))
    rejections.extend(_sweep_positional_construction(resolved_tree))
    rejections.extend(_sweep_rank_divergent_fields(resolved_tree))
    if rejections:
        raise ClosedSchemaError(rejections)
