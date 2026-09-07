"""Tests for the closed helium-importance train schema policy.

The falsifier named in L1a's acceptance contract is "a nested forbidden
reference reaches construction". These tests pin the rejection half of it; the
half that proves nothing was constructed lives with the ``tpen.run`` wiring,
because only there is there anything to construct.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.config_schema import (
    ClosedSchemaError,
    iter_interpolations,
    iter_nodes,
    split_resolver,
)
from tpen.hi_schema import (
    ADMITTED_CALLBACK_TARGETS,
    ADMITTED_METHOD_TARGETS,
    HI_EXPERIMENT_NAME,
    HI_METHOD_ROSTER,
    ADMITTED_UPDATE_METHOD_TARGETS,
    HI_TRAIN_POLICY,
    HI_TRAIN_SCHEMA,
    REFERENCE_MANIFEST_MODULE,
    canonical_train_identity,
    declared_schema,
    is_hi_family,
    validate_hi_train_config,
)


def _config(**sections: object):
    """Return a schema-declaring HI train config with ``sections`` merged in.

    Carries an admitted optimizer by default so that a test about references,
    sections or callbacks is not also a test about method admission. A test
    that means to exercise the method check passes its own ``optimizer``.
    """

    base: dict[str, object] = {
        "schema": HI_TRAIN_SCHEMA,
        "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.005},
    }
    base.update(sections)
    return OmegaConf.create(base)


def _validate(cfg, env: dict[str, str] | None = None) -> None:
    """Validate against an EMPTY launch environment unless one is supplied.

    The real ``os.environ`` is not deterministic across machines, and on a
    cluster login node it carries hundreds of module-system variables. A test
    that used it would pass here and could fail on Cannon for a reason that has
    nothing to do with the case under test.
    """

    validate_hi_train_config(cfg, env={} if env is None else env)


# Anchored to this file, not to the process working directory. The tests above
# use bare relative paths, which pins them to being run from the repository
# root; from anywhere else they read a different tree or none at all. Fixing
# that everywhere is outside this slice, so the new cases below at least do not
# add to it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTROL_CONFIG = _REPO_ROOT / "experiments/atomistic/he-importance/configs/train.yaml"


def _control_with_literal_runner_copy(section: str):
    """Load the control config and give ``runner.<section>`` its own copy.

    The shipped config writes ``runner.model: ${model}``, so the runner's view
    and the root are the same node and no divergence is expressible. Replacing
    the interpolation with a literal copy of the resolved section is what makes
    the two independently editable -- which is the shape the finding is about.

    Returns
    -------
    tuple
        ``(cfg, section_node)``, where mutating ``section_node`` leaves every
        root field compliant.
    """

    cfg = OmegaConf.load(_CONTROL_CONFIG)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg.runner[section] = OmegaConf.create(resolved[section])
    return cfg, cfg.runner[section]


def _rules(error: ClosedSchemaError) -> set[str]:
    return {rejection.rule for rejection in error.rejections}


def _paths(error: ClosedSchemaError) -> set[str]:
    return {rejection.path for rejection in error.rejections}


class TestScope:
    """Who the firewall applies to.

    Not "whoever asks for it": a helium-importance config is REQUIRED to ask,
    and omission is refused in ``TestOmittingTheSchemaKeyIsLoud`` below. These
    cases cover the other side -- a config outside the family passes through,
    which is how the frozen historical fixtures survive without an exemption
    list.
    """

    def test_a_non_family_config_with_no_schema_is_not_validated(self) -> None:
        cfg = OmegaConf.create({"system": {"reference_energy": -2.9}})
        _validate(cfg)

    def test_a_non_family_config_declaring_another_schema_is_not_validated(self) -> None:
        cfg = OmegaConf.create({"schema": "other.v1", "system": {"reference_energy": -2.9}})
        _validate(cfg)

    def test_declared_schema_reads_the_top_level_key(self) -> None:
        assert declared_schema(_config()) == HI_TRAIN_SCHEMA
        assert declared_schema(OmegaConf.create({})) is None

    def test_the_frozen_he_v1_train_fixture_is_left_alone(self) -> None:
        """The historical fixture is noncompliant BY RECORD and must stay loadable.

        ``experiments/atomistic/he-v1/configs/train.yaml`` carries
        ``system.reference_energy``. The plan of record designates it a
        historical implementation fixture to be preserved, not edited. It
        declares no schema, so the firewall must not touch it -- and it must
        also be genuinely noncompliant, or this test would pass for the wrong
        reason and stop protecting anything.
        """

        cfg = OmegaConf.load("experiments/atomistic/he-v1/configs/train.yaml")
        _validate(cfg)

        # Red arm: opting the same content in must reject it.
        opted_in = OmegaConf.merge(OmegaConf.create({"schema": HI_TRAIN_SCHEMA}), cfg)
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(opted_in)
        assert "system.reference_energy" in _paths(caught.value)


class TestOmittingTheSchemaKeyIsLoud:
    """The marker is mandatory for this family, so its ABSENCE must fail.

    An opt-in firewall with no detector for omission is fail-open: a new
    helium-importance train config that forgets ``schema:`` would get zero
    enforcement and nothing anywhere would go red. These tests are that
    detector.
    """

    def test_an_hi_config_without_the_schema_key_is_refused(self) -> None:
        cfg = OmegaConf.create(
            {
                "experiment": {"name": HI_EXPERIMENT_NAME},
                "system": {"reference_energy": -2.9},
            }
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-schema" in _rules(caught.value)

    def test_a_clean_hi_config_without_the_key_is_still_refused(self) -> None:
        """Refused for the OMISSION itself, not because it also has a reference.

        Without this case the rule would look satisfied by a test that a
        reference-bearing config gets rejected -- which it would be anyway once
        the key is present. The omission has to be the sole cause.
        """

        cfg = OmegaConf.create({"experiment": {"name": HI_EXPERIMENT_NAME}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert _rules(caught.value) == {"undeclared-schema"}

    def test_an_hi_config_declaring_the_wrong_schema_is_refused(self) -> None:
        cfg = OmegaConf.create(
            {"schema": "tpen.hi.evaluation.v1", "experiment": {"name": HI_EXPERIMENT_NAME}}
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-schema" in _rules(caught.value)

    def test_a_broken_interpolation_does_not_exempt_an_hi_config(self) -> None:
        """Family membership is read WITHOUT resolving, on purpose.

        If it resolved, an unrelated typo elsewhere in the file would make the
        config unrecognisable and silently downgrade it to unenforced -- the
        finding and the thing that hides it sharing a failure domain again.
        """

        cfg = OmegaConf.create(
            {"experiment": {"name": HI_EXPERIMENT_NAME}, "model": {"c": "${nope.missing}"}}
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-schema" in _rules(caught.value)

    @pytest.mark.parametrize("name", ["tpen_he_v1", "tpen_h2_v1", "tpen_pair_v1"])
    def test_another_experiment_is_untouched(self, name: str) -> None:
        """The three other experiment families a default-ON rule would have broken."""

        cfg = OmegaConf.create(
            {"experiment": {"name": name}, "system": {"reference_energy": -2.9}}
        )
        _validate(cfg)

    def test_the_frozen_he_v1_fixture_needs_no_exemption_entry(self) -> None:
        """It survives by NOT being in the family, not by being listed."""

        cfg = OmegaConf.load("experiments/atomistic/he-v1/configs/train.yaml")
        assert not is_hi_family(cfg)
        _validate(cfg)

    def test_the_control_config_is_recognised_as_family(self) -> None:
        """Positive control: if nothing were in the family, the rule is inert."""

        cfg = OmegaConf.load("experiments/atomistic/he-importance/configs/train.yaml")
        assert is_hi_family(cfg)


class TestForbiddenSurfaces:
    def test_rejects_a_nested_reference_energy(self) -> None:
        cfg = _config(system={"nuclei": {"reference_energy": -2.903724377034119598}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-surface:reference" in _rules(caught.value)
        assert "system.nuclei.reference_energy" in _paths(caught.value)

    def test_rejects_a_reference_reached_only_through_interpolation(self) -> None:
        """The forbidden value is a plain number until resolution moves it.

        ``model.exact`` is not a forbidden key name. What makes this a violation
        is that resolution places the value under ``trainer.baseline_energy``,
        which is. A raw-tree-only sweep would still catch the destination key
        here; the point of the case is that the two trees agree, so a later
        change that drops one sweep does not silently pass this config.
        """

        cfg = _config(
            model={"scalar": -2.9},
            trainer={"baseline_energy": "${model.scalar}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        trees = {r.tree for r in caught.value.rejections if r.path == "trainer.baseline_energy"}
        assert trees == {"raw", "resolved"}

    @pytest.mark.parametrize(
        ("key", "rule"),
        [
            ("energy_gap", "forbidden-surface:gap"),
            ("accuracy_band", "forbidden-surface:band"),
            ("continuation_from", "forbidden-surface:continuation"),
            ("ground_truth", "forbidden-surface:reference"),
            # A stop rule keyed to a target names no reference and would pass
            # every other family; ``stop`` is what makes it visible.
            ("stop_at_energy", "forbidden-surface:stop-rule"),
            # Both spellings, because they tokenize differently: ``stop`` and
            # ``stopping`` are separate tokens under whole-token matching.
            ("early_stop", "forbidden-surface:stop-rule"),
            ("early_stopping", "forbidden-surface:stop-rule"),
            ("patience", "forbidden-surface:stop-rule"),
        ],
    )
    def test_rejects_each_forbidden_family(self, key: str, rule: str) -> None:
        cfg = _config(trainer={key: 1})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert rule in _rules(caught.value)

    def test_checkpoint_resume_is_not_a_continuation_surface(self) -> None:
        """Resume is recovery; continuation is selection. Only one is forbidden.

        ``tpen.checkpoint.TrainResume`` is the standard payload and appears in
        every training configuration. A continuation rule that also swept
        ``resume`` would make the schema unsatisfiable.
        """

        cfg = _config(
            callbacks=[
                {
                    "_target_": "tpen.callback.Checkpoint",
                    "payload": {"_target_": "tpen.checkpoint.TrainResume"},
                    "output_dir": "out",
                }
            ]
        )
        _validate(cfg)

    def test_bandwidth_is_not_a_band_surface(self) -> None:
        """Substring matching would reject this ordinary key."""

        _validate(_config(sampler={"bandwidth": 1.0}))

    def test_backstop_is_not_a_stop_rule_surface(self) -> None:
        """Substring matching would reject this ordinary key."""

        _validate(_config(trainer={"backstop": 1, "nonfinite_local_energy_policy": "fail"}))

    def test_the_stop_rule_surface_can_fire_on_a_surface_that_does_not_exist_yet(self) -> None:
        """The forward guard trips on a NEW key, not only on ones written today.

        No module under ``tpen/`` implements a stop rule, so every rejection
        above is a rule aimed at a surface nobody has built. That makes one
        failure mode specific to this family: a guard that only recognises the
        exact key names its author imagined is decoration, because whoever adds
        stopping tomorrow will name it something else and the guard will stay
        green while the hazard lands.

        These key names were chosen to be ones NOT enumerated in the
        parametrized list above, and they are placed in three different
        sections, so the family is shown to be keyed on the token rather than
        on a remembered spelling or a remembered location.
        """

        for section, key in (
            ("trainer", "halt_when_converged"),
            ("runner", "stopping_criterion"),
            ("sampler", "patience_steps"),
        ):
            cfg = _config(**{section: {key: 1}})
            with pytest.raises(ClosedSchemaError) as caught:
                _validate(cfg)
            assert "forbidden-surface:stop-rule" in _rules(caught.value), (
                f"{section}.{key} did not trip the stop-rule family; the guard "
                "recognises only spellings someone thought of in advance"
            )


class TestFrozenScalarsAreCheckedWhereTheyAreCONSUMED:
    """A dotted rule checked where the value is DECLARED, not where it is used.

    ``system.spatial_dim`` and ``runtime.dtype`` are the study's canonical
    declarations, but components are built from ``model.embedding.spatial_dim``,
    ``sampler.spatial_dim`` and ``sampler.dtype``. The shipped config wires
    those with interpolations, and nothing required it to.

    MEASURED: with ``system.spatial_dim: 3`` left compliant, each of those three
    consumption sites accepted a divergent value, and the model would have been
    built in TWO dimensions. That is F5's own shape -- the schema validating
    ROOT fields while what is actually constructed carries its own copy --
    applied to a scalar instead of a component, which is the finding this slice
    exists for.

    FOUND BY THE AUTHOR while auditing what remained anchored after the reviewer
    showed the class was still open twice over.
    """

    @pytest.mark.parametrize(
        ("dotted", "value"),
        [
            ("model.embedding.spatial_dim", 2),
            ("sampler.spatial_dim", 2),
            ("sampler.dtype", "float32"),
        ],
    )
    def test_a_divergent_value_at_a_consumption_site_is_refused(
        self, dotted: str, value: object
    ) -> None:
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        OmegaConf.update(cfg, dotted, value)
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert dotted in _paths(caught.value)

    def test_the_canonical_declaration_is_still_checked(self) -> None:
        """Non-regression: widening must not lose the case the rule had."""

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.system.spatial_dim = 2
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "system.spatial_dim" in _paths(caught.value)

    def test_an_absent_coordinate_is_not_a_violation(self) -> None:
        """Absence inherits a correct default; only a VALUE is refused.

        A rule that refused the KEY rather than a value would refuse every
        component that simply does not mention the coordinate.
        """

        _validate(_config(run={"run_id": "x"}, model={"embedding": {"out_channels": 32}}))

    def test_the_shipped_config_declares_these_consistently(self) -> None:
        """The over-restriction measurement, asserted rather than remembered.

        This rule refuses a divergent ``spatial_dim`` or ``dtype`` ANYWHERE. It
        is only defensible while every such node in a shipped config resolves to
        the study value, so that is a test rather than a comment.
        """

        resolved = OmegaConf.to_container(OmegaConf.load(_CONTROL_CONFIG), resolve=True)
        seen = {
            path: value
            for path, key, value in iter_nodes(resolved)
            if key in ("spatial_dim", "dtype")
        }
        assert seen, "no frozen-scalar nodes found; this test would pass vacuously"
        for path, value in seen.items():
            assert value in (3, "float64"), (path, value)


class TestTheTrainerRulesSurviveAFactoryWrapper:
    """`trainer.update_method` and the policy were anchored to ``trainer``.

    With the trainer written as a Hydra factory wrapper, the rules found nothing
    on the wrapper and returned while the real spec sat one level down.

    THE TRAP WORTH RECORDING is that this first LOOKED closed. The wrapper also
    hid ``nonfinite_local_energy_policy``, so a DIFFERENT rule fired and the
    config was refused for an unrelated reason. Restating the policy on the
    wrapper removed that accident and an unadmitted
    ``StochasticReconfigurationUpdate`` validated. **A refusal is only evidence
    for the rule that produced it.**
    """

    SR = {"_target_": "tpen.training.sr.StochasticReconfigurationUpdate", "_partial_": True}
    LEGACY = {"_target_": "tpen.training.update.LegacyAutogradUpdate", "_partial_": True}

    def _wrapped_trainer(self, mutate, wrapper_extra=None):
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
        mutate(trainer)
        body = {"_target_": "hydra.utils.instantiate", "_recursive_": False, "config": trainer}
        if wrapper_extra:
            body.update(wrapper_extra)
        cfg.trainer = OmegaConf.create(body)
        return cfg

    def test_an_unadmitted_update_rule_behind_the_wrapper_is_refused(self) -> None:
        cfg = self._wrapped_trainer(
            lambda tr: tr.__setitem__("update_method", self.SR),
            {"nonfinite_local_energy_policy": "fail"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-update-method" in _rules(caught.value), (
            "must be refused BY THE UPDATE-METHOD RULE; a refusal from any other "
            "rule is the accident this test exists to rule out"
        )

    def test_an_admitted_update_rule_behind_the_wrapper_validates(self) -> None:
        """The over-restriction control: the wrapper itself is admissible."""

        _validate(
            self._wrapped_trainer(
                lambda tr: tr.__setitem__("update_method", self.LEGACY),
                {"nonfinite_local_energy_policy": "fail"},
            )
        )

    def test_an_inadmissible_policy_behind_the_wrapper_is_refused(self) -> None:
        """The inner policy is what the constructed trainer uses."""

        cfg = self._wrapped_trainer(
            lambda tr: tr.__setitem__("nonfinite_local_energy_policy", "drop"),
            {"nonfinite_local_energy_policy": "fail"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-nonfinite-policy" in _rules(caught.value)

    def test_an_omitted_policy_is_still_refused(self) -> None:
        """Non-regression on the required-declaration half of that rule."""

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        del cfg.trainer.nonfinite_local_energy_policy
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-nonfinite-policy" in _rules(caught.value)


class TestTheReadoutRuleHasTwoNets:
    """A path rule was escapable by INDIRECTION, so a shape net was added.

    Hydra's own ``hydra.utils.instantiate`` can be a ``_target_``::

        model:
          _target_: hydra.utils.instantiate
          _recursive_: false
          config: { ...the real model, readout frozen... }

    That constructs the same model with the whole subtree one level down, so
    ``model.readout.trainable`` names nothing. MEASURED by an independent
    verifier: this validated and constructed a real ``TPENWaveFunction`` whose
    ``PfaffianReadout`` had ZERO parameters and forwarded finitely -- a run that
    trains for its whole budget with a permanently frozen readout, which is the
    exact he-v1 defect this rule exists to prevent.

    THE DIAGNOSIS IS NARROWER THAN "BAN THE WRAPPER". Through the SAME wrapper
    an unadmitted cusp law, a moved ``max_order``, an omitted non-finite policy
    and an unadmitted optimizer were all still refused, because those rules
    identify a thing by what it IS or by key at any depth. The indirection only
    defeated the last rule that identified a thing by where it SITS.
    """

    def _wrapped(self, mutate=None):
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        model = OmegaConf.to_container(cfg.model, resolve=True)
        if mutate is not None:
            mutate(model)
        cfg.model = OmegaConf.create(
            {"_target_": "hydra.utils.instantiate", "_recursive_": False, "config": model}
        )
        return cfg

    def test_a_frozen_readout_behind_the_indirection_is_refused(self) -> None:
        cfg = self._wrapped(lambda m: m["readout"].__setitem__("trainable", False))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.config.readout.trainable" in _paths(caught.value)

    def test_an_omitted_declaration_behind_the_indirection_is_refused(self) -> None:
        cfg = self._wrapped(lambda m: m["readout"].pop("trainable"))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-trainability" in _rules(caught.value)

    def test_the_indirection_is_closed_by_DEPTH_not_by_target_matching(self) -> None:
        """Attribution, pinned, because the obvious reading is wrong.

        The wrapped readout still arrives under the key ``readout``, at
        ``model.config.readout``. What closed the escape is the key net walking
        the model subtree at ANY DEPTH -- not the shape net. A mutant removing
        the shape net left the wrapped case refused and changed no behaviour on
        that probe, which is how the misattribution was caught before it
        shipped.

        This test states the property the fix actually relies on, so a later
        reader optimising the shape net away does not believe they are removing
        what holds the indirection closed.
        """

        cfg = self._wrapped(lambda m: m["readout"].__setitem__("trainable", False))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        offending = {r.path for r in caught.value.rejections if "trainab" in r.rule}
        assert "model.config.readout.trainable" in offending
        # The key is unchanged by the wrapper; only the DEPTH changed.
        assert all(path.endswith(".readout.trainable") for path in offending), offending

    def test_the_shape_net_catches_a_readout_under_another_key(self) -> None:
        """The case only the SHAPE net catches, recorded with its real severity.

        A ``PfaffianReadout`` under a key that is not ``readout`` is invisible to
        the key net. This is a PRECONSTRUCTION gap rather than a live escape --
        ``TPENWaveFunction`` takes ``**kwargs`` so the key is swallowed, and
        ``readout`` is a required keyword, so moving the readout there fails at
        construction. Closed anyway, for the reason F2's SR case was: "another
        component happens to refuse it" is a property of today's constructors,
        not a rule.

        Without this arm the shape net is unguarded -- a mutant removing it
        passes the whole suite, which is exactly what happened before this test
        existed.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        readout = OmegaConf.to_container(cfg.model.readout, resolve=True)
        readout["trainable"] = False
        OmegaConf.update(cfg, "model.head", readout, force_add=True)
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.head.trainable" in _paths(caught.value)

    def test_a_compliant_model_behind_the_indirection_validates(self) -> None:
        """The over-restriction control: refuse the FREEZE, not the wrapper.

        Banning ``hydra.utils.instantiate`` outright would pass both red arms
        above while forbidding a Hydra feature that, as measured, defeats none
        of the shape-based rules.
        """

        _validate(self._wrapped())

    def test_the_key_net_still_catches_a_readout_with_no_target(self) -> None:
        """The net that was KEPT, and why keeping it was not sentimentality.

        A readout written without a ``_target_`` matches no shape rule. Unlike a
        Hamiltonian term -- which ``_validate_hamiltonian_term`` refuses loudly
        for lacking a callable ``local_energy`` -- a bare readout mapping has no
        comparably crisp construction-time backstop, so dropping the key net to
        make the design uniform would have removed real coverage.
        """

        cfg = _config(model={"readout": {"channels": 32, "trainable": False}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-trainability" in _rules(caught.value)

    def test_one_readout_is_not_reported_twice_by_the_two_nets(self) -> None:
        """The shipped readout matches BOTH nets; it must yield one finding.

        Two rejections for one field read as two defects. De-duplication is by
        PATH, so a genuinely second readout elsewhere is still reported.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.model.readout.trainable = False
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        hits = [r for r in caught.value.rejections if r.path == "model.readout.trainable"]
        assert len(hits) == 1, [r.path for r in caught.value.rejections]


class TestHamiltonianTermsAreFoundByShapeNotByKey:
    """The terms may be a Mapping OR a Sequence, and the keys are the author's.

    ``normalize_hamiltonian_terms`` accepts either, and a sequence falls back to
    snake-case class names. So a dotted rule on
    ``hamiltonian_terms.electron_nucleus.eps`` checks a NAME the config author
    was never obliged to use.

    MEASURED BEFORE THE CHANGE: with ``eps: 0.01`` the mapping form spelled
    ``electron_nucleus`` was correctly refused, while the same Hamiltonian
    written as a SEQUENCE, or as a mapping keyed ``en``, VALIDATED. No
    ``_args_``, no wrapper -- ordinary keyword configuration.

    The floor is not cosmetic: a nonzero one masks the near-nucleus
    cancellation the local-energy qualification has to measure.

    FOUND BY THE AUTHOR, not by the reviewer, while auditing whether the
    name-shaped CLASS was closed after four instances of it had been fixed. It
    was not. That is the answer to the question, and it is recorded here rather
    than in a commit message because the next person to add a dotted rule to
    this module needs it.
    """

    def _terms(self, shape: str, eps: float):
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
        terms["electron_nucleus"]["eps"] = eps
        if shape == "mapping":
            cfg.hamiltonian_terms = OmegaConf.create(terms)
        elif shape == "renamed":
            terms["en"] = terms.pop("electron_nucleus")
            cfg.hamiltonian_terms = OmegaConf.create(terms)
        elif shape == "sequence":
            cfg.hamiltonian_terms = OmegaConf.create(
                [
                    terms["kinetic"],
                    terms["electron_nucleus"],
                    terms["electron_electron"],
                    terms["nucleus_nucleus"],
                ]
            )
        else:  # pragma: no cover - guards a typo in a parametrization
            raise AssertionError(shape)
        return cfg

    # Spelled out rather than derived from the rule, so narrowing the rule
    # cannot silently remove the arm that proves the escape is closed.
    SHAPES = ["mapping", "renamed", "sequence"]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_a_nonzero_coulomb_floor_is_refused_in_any_shape(self, shape: str) -> None:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(self._terms(shape, 0.01))
        assert "frozen-coordinate" in _rules(caught.value)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_study_floor_validates_in_any_shape(self, shape: str) -> None:
        """The over-restriction control: refuse the VALUE, never the shape."""

        _validate(self._terms(shape, 0.0))

    @pytest.mark.parametrize(
        ("shape", "expected_path"),
        [
            ("mapping", "hamiltonian_terms.electron_nucleus.eps"),
            ("renamed", "hamiltonian_terms.en.eps"),
            ("sequence", "hamiltonian_terms[1].eps"),
        ],
    )
    def test_the_reported_path_names_where_the_term_actually_is(
        self, shape: str, expected_path: str
    ) -> None:
        """A path assuming one shape points at nothing in the other two."""

        with pytest.raises(ClosedSchemaError) as caught:
            _validate(self._terms(shape, 0.01))
        assert expected_path in _paths(caught.value)

    def test_an_omitted_floor_is_not_a_violation(self) -> None:
        """Absence applies the constructor default, and that default IS 0.0.

        Requiring the declaration would refuse configs that simply omit it,
        which is the frozen-scalar precedent rather than the trainability one --
        the distinction being whether the inherited value is the right one.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        del cfg.hamiltonian_terms.electron_nucleus.eps
        _validate(cfg)

    def test_a_term_without_a_target_is_not_constructible_anyway(self) -> None:
        """The NARROWING this change makes, stated and pinned rather than hidden.

        The old dotted rule fired on any node at
        ``hamiltonian_terms.electron_nucleus.eps``, including one with no
        ``_target_``. The shape-based rule identifies a term by its target, so
        such a node is no longer matched. That loses no real coverage, and this
        asserts WHY rather than asking a reader to trust it:
        ``normalize_hamiltonian_terms`` requires every term to expose a callable
        ``local_energy``, so a bare mapping fails loudly at construction.

        A narrowing justified by "the other layer catches it" is exactly the
        claim this slice got wrong about transitive imports, so the other layer
        is exercised here rather than cited.
        """

        pytest.importorskip("torch", reason="tpen.physics.hamiltonian imports torch")
        from tpen.physics.hamiltonian import normalize_hamiltonian_terms

        with pytest.raises(TypeError, match="local_energy"):
            normalize_hamiltonian_terms({"electron_nucleus": {"eps": 1e-8}})

    def test_the_electron_electron_floor_is_deliberately_unpinned(self) -> None:
        """Scope control: this fixed an ESCAPE, it did not add a constraint.

        ``ElectronElectronInteraction`` also takes ``eps`` and also defaults to
        0.0, but no slice has pinned it and its own docstring records the
        finite-eps electron-electron case as UNMEASURED. Pinning it here would
        be a new scientific constraint smuggled in under a bug fix. If a later
        slice decides to pin it, this test is the one to delete, and deleting it
        should require saying why.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.hamiltonian_terms.electron_electron.eps = 0.01
        _validate(cfg)


class TestFactorsAreFoundByShapeNotByContainer:
    """A factor is what its ``_target_`` says, not where the config puts it.

    Both factor rules used to read ``model.factors`` and return unless it was a
    LIST. ``TPENWaveFunction`` accepts any iterable and normalizes it, so a
    ``torch.nn.ModuleList`` block is a valid, constructible configuration in
    which ``model.factors`` is a MAPPING -- and every factor rule skipped it.

    MEASURED BEFORE THE CHANGE: an unadmitted ``CurvatureElectronNucleusCuspLaw``
    and a frozen electron-electron factor both VALIDATED inside that wrapper,
    with no ``_args_`` anywhere. This is the residual the `_args_` family
    refusal did not reach: ordinary keyword configuration, existing public API.
    """

    OLD_LAW = "tpen.nn.CurvatureElectronNucleusCuspLaw"

    def _with_factors(self, container: str, mutate=None):
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        factors = OmegaConf.to_container(cfg.model.factors, resolve=True)
        if mutate is not None:
            mutate(factors)
        if container == "list":
            cfg.model.factors = OmegaConf.create(factors)
        elif container == "modulelist":
            cfg.model.factors = OmegaConf.create(
                {"_target_": "torch.nn.ModuleList", "modules": factors}
            )
        elif container == "nested":
            cfg.model.factors = OmegaConf.create(
                {
                    "_target_": "torch.nn.ModuleList",
                    "modules": {"_target_": "torch.nn.ModuleList", "modules": factors},
                }
            )
        else:  # pragma: no cover - guards a typo in a parametrization
            raise AssertionError(container)
        return cfg

    # The containers are SPELLED OUT rather than read from the schema, so a rule
    # that stopped recognising one would remove the escape without removing the
    # arm that proves it is closed.
    CONTAINERS = ["list", "modulelist", "nested"]

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_an_unadmitted_cusp_law_is_refused_in_any_container(self, container: str) -> None:
        cfg = self._with_factors(
            container, lambda f: f[1]["law"].__setitem__("_target_", self.OLD_LAW)
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-cusp-law" in _rules(caught.value)

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_frozen_factor_is_refused_in_any_container(self, container: str) -> None:
        cfg = self._with_factors(container, lambda f: f[0].__setitem__("trainable_range", False))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-trainability" in _rules(caught.value)

    @pytest.mark.parametrize("container", CONTAINERS)
    def test_a_compliant_configuration_validates_in_any_container(self, container: str) -> None:
        """The over-restriction control, one arm per container.

        The rule must refuse the DIVERGENCE, not the wrapper. Refusing
        ``ModuleList`` outright would pass both red tests above while
        forbidding a legitimate way to write the model.
        """

        _validate(self._with_factors(container))

    def test_the_reported_path_names_where_the_factor_actually_IS(self) -> None:
        """A path that assumes the list shape sends the reader to nothing.

        The old rule hard-coded ``model.factors[i]``. Inside a wrapper the
        factor lives at ``model.factors.modules[i]``, and a rejection naming the
        first would point at a key that does not exist.
        """

        cfg = self._with_factors(
            "modulelist", lambda f: f[1]["law"].__setitem__("_target_", self.OLD_LAW)
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.factors.modules[1].law._target_" in _paths(caught.value)

    def test_a_factor_class_outside_the_model_is_not_given_a_factor_rule(self) -> None:
        """Scoped to ``model``: a factor is a factor where the model is built.

        The same class named in a diagnostic's arguments is not the model's
        factor and carries no trainability contract. Without this the rule would
        reach into unrelated sections and refuse them for missing a declaration
        they never owed.
        """

        _validate(
            _config(
                run={"run_id": "x"},
                sampler={
                    "_target_": "tpen.sampling.metropolis.MetropolisSampler",
                    "warmup_probe": {"_target_": "tpen.nn.ElectronElectronCusp"},
                },
            )
        )


class TestPositionalConstructionIsRefused:
    """`_args_` builds the same components with no key for any rule to match.

    Every component rule identifies what it judges by the KEY the component
    hangs from. ``tpen.runner.Train`` takes
    ``(model, sampler, hamiltonian_terms, optimizer, trainer)`` positionally, so
    ``runner._args_`` constructs all five with an index instead of a name.

    MEASURED BEFORE THE RULE EXISTED: all five divergences the component views
    refuse by keyword were ACCEPTED when the same content was passed
    positionally. This is the key-versus-value failure that produced the
    target-allowlist gap, one level up.
    """

    def _positional_runner(self, mutate=None):
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        args = [
            resolved["model"],
            resolved["sampler"],
            resolved["hamiltonian_terms"],
            resolved["optimizer"],
            resolved["trainer"],
        ]
        if mutate is not None:
            mutate(args)
        cfg.runner = OmegaConf.create({"_target_": "tpen.runner.Train", "_args_": args})
        return cfg

    def test_an_unmodified_positional_runner_is_refused(self) -> None:
        """Refused even when it diverges in NO way.

        This is the point of a family refusal: the schema cannot tell a benign
        positional runner from a divergent one, because it cannot tell which
        slot is which. Admitting the benign case would require exactly the
        name-matching that positional construction removes.
        """

        with pytest.raises(ClosedSchemaError) as caught:
            _validate(self._positional_runner())
        assert "positional-construction" in _rules(caught.value)
        assert "runner._args_" in _paths(caught.value)

    @pytest.mark.parametrize(
        ("label", "mutate"),
        [
            ("frozen readout", lambda a: a[0]["readout"].__setitem__("trainable", False)),
            ("global clip", lambda a: a[4].__setitem__("gradient_clip_norm", 1.0)),
            ("omitted nonfinite policy", lambda a: a[4].pop("nonfinite_local_energy_policy")),
            (
                "old cusp law",
                lambda a: a[0]["factors"][1]["law"].__setitem__(
                    "_target_", "tpen.nn.CurvatureElectronNucleusCuspLaw"
                ),
            ),
            ("unadmitted optimizer", lambda a: a[3].__setitem__("_target_", "torch.optim.SGD")),
        ],
    )
    def test_every_divergence_that_escaped_positionally_is_now_refused(
        self, label: str, mutate
    ) -> None:
        """One arm per divergence, each measured as ACCEPTED before the rule.

        Read as a set these would be one assertion naming none of them; the
        point of five arms is that a later narrowing of the rule says which
        escape it reopened.
        """

        with pytest.raises(ClosedSchemaError) as caught:
            _validate(self._positional_runner(mutate))
        assert "positional-construction" in _rules(caught.value), label

    def test_positional_construction_is_refused_outside_the_runner_too(self) -> None:
        """The weakness is path-shaped rules generally, not the runner section.

        A readout passed positionally under ``model`` defeats
        ``model.readout.trainable`` exactly as a trainer passed positionally
        under ``runner`` defeats ``trainer.gradient_clip_norm``.
        """

        cfg = _config(
            run={"run_id": "x"},
            model={"_target_": "tpen.nn.TPENWaveFunction", "_args_": [{"channels": 32}]},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model._args_" in _paths(caught.value)

    def test_the_shipped_configs_use_no_positional_construction(self) -> None:
        """The over-restriction measurement, asserted rather than remembered.

        The rule refuses a Hydra feature outright. That is only defensible while
        nothing shipped uses it, so the claim is a test rather than a comment --
        a comment would keep saying so after it stopped being true.
        """

        for path in sorted(_CONTROL_CONFIG.parent.glob("*.yaml")):
            assert "_args_" not in path.read_text(encoding="utf-8"), path
            _validate(OmegaConf.load(path))


class TestTheGradientClipKnobIsCheckedByKey:
    """Clipping is not owned by the trainer, so a dotted path cannot bound it.

    ``LegacyAutogradUpdate.__init__`` takes ``gradient_clip_norm`` and is the
    object that APPLIES it. An ``update_method`` block naming that admitted
    class with a clip therefore clipped every update while
    ``trainer.gradient_clip_norm`` sat compliantly at null -- the rule checked
    the field the study happens to spell, not the knob.
    """

    def test_the_admitted_update_method_may_not_carry_a_clip(self) -> None:
        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.trainer.update_method = OmegaConf.create(
            {
                "_target_": "tpen.training.update.LegacyAutogradUpdate",
                "_partial_": True,
                "gradient_clip_norm": 1.0,
            }
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "trainer.update_method.gradient_clip_norm" in _paths(caught.value)

    def test_the_canonical_trainer_path_is_still_refused(self) -> None:
        """Non-regression: widening the rule must not lose the case it had."""

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.trainer.gradient_clip_norm = 2.0
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "trainer.gradient_clip_norm" in _paths(caught.value)

    def test_an_explicit_null_clip_on_the_update_method_is_admitted(self) -> None:
        """The over-restriction control: null and absent both say NO clipping.

        The control config writes ``gradient_clip_norm: null`` deliberately, to
        state the policy rather than inherit it. A rule that refused the KEY
        rather than a VALUE would refuse the shipped config itself.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.trainer.update_method = OmegaConf.create(
            {
                "_target_": "tpen.training.update.LegacyAutogradUpdate",
                "_partial_": True,
                "gradient_clip_norm": None,
            }
        )
        _validate(cfg)

    def test_an_interpolated_clip_is_reported_at_every_path_it_reaches(self) -> None:
        """One CONFIGURED clip, two RESOLVED paths, and both are named.

        The control writes ``runner.trainer: ${trainer}``, so the resolved tree
        contains the field twice and the sweep reports it twice. That is the
        honest behaviour, not a defect: when a runner carries a LITERAL copy
        rather than an interpolation those are two independent fields, and a
        rule that de-duplicated by value would name only one and send the reader
        to the wrong place.

        Written after asserting the opposite. The first version of this test
        claimed one rejection, on the theory that sweeping whole-tree instead of
        per-view removed a duplicate. It does not -- interpolation duplicates
        the CONTENT, so both traversals see two -- and the docstring on
        `_sweep_gradient_clip` was corrected with it.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        cfg.trainer.gradient_clip_norm = 2.0
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        clips = {r.path for r in caught.value.rejections if r.path.endswith("gradient_clip_norm")}
        assert clips == {"trainer.gradient_clip_norm", "runner.trainer.gradient_clip_norm"}


class TestAdmittedUpdateMethods:
    """The optimizer roster qualifies ``optimizer._target_`` and nothing else.

    A configuration selects its update RULE at ``trainer.update_method`` and its
    optimizer at ``optimizer``. Those are two surfaces and only the second was
    qualified, so a config could name Adam -- admitted, roster-clean -- and an
    unadmitted update rule beside it.

    SEVERITY, RECORDED HONESTLY: the observed example is a PRECONSTRUCTION gap
    rather than a successful unadmitted run, because SR with Adam is refused
    later by the SR constructor. "Some other component happens to refuse it" is
    a property of today's constructors and not a rule, which is why it is closed
    here rather than left to them.
    """

    SR_UPDATE = "tpen.training.sr.StochasticReconfigurationUpdate"

    def _trainer(self, **extra: object) -> dict[str, object]:
        return {
            "_target_": "tpen.training.trainer.VMCTrainer",
            "nonfinite_local_energy_policy": "fail",
            **extra,
        }

    def test_the_roster_and_the_update_allowlist_are_different_surfaces(self) -> None:
        """Pins the SCOPE claim rather than the prose that states it.

        A reader meeting ``ADMITTED_METHOD_TARGETS`` has to know it governs
        optimizers only. Asserting the two sets are disjoint, and that SR's
        update class is in neither, is what keeps that true if either set moves.
        """

        assert ADMITTED_METHOD_TARGETS.isdisjoint(ADMITTED_UPDATE_METHOD_TARGETS)
        assert self.SR_UPDATE not in ADMITTED_METHOD_TARGETS
        assert self.SR_UPDATE not in ADMITTED_UPDATE_METHOD_TARGETS

    def test_rejects_an_unadmitted_update_rule_beside_an_admitted_optimizer(self) -> None:
        """The finding's exact shape: the roster passes and the rule does not."""

        cfg = _config(
            run={"run_id": "x"},
            trainer=self._trainer(update_method={"_target_": self.SR_UPDATE, "_partial_": True}),
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-update-method" in _rules(caught.value)
        assert "unadmitted-method" not in _rules(caught.value), (
            "the optimizer roster must NOT have fired -- Adam is admitted. If it did, "
            "this arm is measuring the wrong rule"
        )
        assert "trainer.update_method._target_" in _paths(caught.value)

    def test_the_refusal_states_what_the_roster_says(self) -> None:
        cfg = _config(
            run={"run_id": "x"},
            trainer=self._trainer(update_method={"_target_": self.SR_UPDATE}),
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        detail = " ".join(r.detail for r in caught.value.rejections)
        assert "EXCLUDED from the helium-importance scan" in detail
        assert "optimizer._target_ only" in detail

    def test_rejects_an_update_method_that_declares_no_target(self) -> None:
        cfg = _config(
            run={"run_id": "x"},
            trainer=self._trainer(update_method={"damping": 0.001}),
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "trainer.update_method._target_" in _paths(caught.value)

    def test_rejects_an_update_method_that_is_not_a_config_block(self) -> None:
        cfg = _config(run={"run_id": "x"}, trainer=self._trainer(update_method="sr"))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "trainer.update_method" in _paths(caught.value)

    def test_rejects_an_unadmitted_update_rule_in_the_runner_view(self) -> None:
        """The two fixes compose: an unqualified rule under `runner` is caught."""

        cfg, trainer = _control_with_literal_runner_copy("trainer")
        trainer.update_method = OmegaConf.create({"_target_": self.SR_UPDATE})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.trainer.update_method._target_" in _paths(caught.value)

    def test_an_omitted_update_method_is_admitted(self) -> None:
        """What every shipped configuration does.

        Absence resolves to `LegacyAutogradUpdate`, the plain optimizer step,
        which IS the admitted Adam method. Requiring the declaration would
        refuse the control config.
        """

        _validate(_config(run={"run_id": "x"}, trainer=self._trainer()))

    def test_an_explicitly_null_update_method_is_admitted(self) -> None:
        _validate(_config(run={"run_id": "x"}, trainer=self._trainer(update_method=None)))

    @pytest.mark.parametrize(
        "target",
        [
            # SPELLED OUT, NOT DRAWN FROM THE SET UNDER TEST. Parametrizing over
            # `ADMITTED_UPDATE_METHOD_TARGETS` made the arms vary WITH the
            # subject: removing a spelling from the allowlist removed the arm
            # that would have caught it, and a mutant that dropped the alias
            # left the whole suite green. The arms have to vary independently of
            # the thing they measure or they measure nothing.
            "tpen.training.update.LegacyAutogradUpdate",
            "tpen.training.LegacyAutogradUpdate",
        ],
    )
    def test_accepts_every_admitted_spelling(self, target: str) -> None:
        """One arm per spelling, because `tpen.training` re-exports the class.

        Hydra resolves either path to the same object, so an allowlist naming
        one would refuse a configuration that is correct -- an over-restriction
        that surfaces as a run that cannot start.
        """

        assert target in ADMITTED_UPDATE_METHOD_TARGETS
        _validate(
            _config(
                run={"run_id": "x"},
                trainer=self._trainer(update_method={"_target_": target, "_partial_": True}),
            )
        )

    def test_the_admitted_spellings_name_the_same_class(self) -> None:
        """Guards the allowlist against an entry that resolves nowhere.

        Two strings in a set look equally valid; only importing them shows that
        both name the class the trainer actually falls back to. Skipped where
        torch is absent, because importing `tpen.training` needs it.
        """

        torch_backed = pytest.importorskip("tpen.training", reason="needs torch")
        from importlib import import_module

        resolved = set()
        for target in ADMITTED_UPDATE_METHOD_TARGETS:
            module_path, _, attribute = target.rpartition(".")
            resolved.add(getattr(import_module(module_path), attribute))
        assert len(resolved) == 1
        assert resolved == {torch_backed.LegacyAutogradUpdate}


class TestTheRunnerViewIsValidatedToo:
    """What Hydra constructs is `cfg.runner`, and `instantiate` is recursive.

    Every component rule reads a path from the configuration ROOT. A runner
    section carrying its own literal copies was therefore accepted with a
    frozen readout, a global gradient clip, an unadmitted cusp law and an
    omitted non-finite policy, while every root field stayed compliant. This is
    not a new class of rule failing -- it is the SAME rules, applied to what is
    actually built.

    THE PATH IS PART OF EACH ASSERTION. A finding reported at
    ``trainer.gradient_clip_norm`` when the offending value is at
    ``runner.trainer.gradient_clip_norm`` sends the reader to a compliant
    field, and asserting only the rule name would not tell the two apart.
    """

    def test_the_control_config_still_validates(self) -> None:
        """The over-restriction control, run FIRST because it is load-bearing.

        The shipped config carries `runner.model: ${model}`, so every component
        is now swept twice. If the second pass refused anything the real run
        would not start.
        """

        _validate(OmegaConf.load(_CONTROL_CONFIG))

    def test_a_runner_copy_may_still_equal_the_root(self) -> None:
        """A literal copy that DIVERGES IN NO WAY is admissible.

        Root-equality was the other candidate remedy and would also have passed
        every red arm below, by refusing any runner section that is not the
        root. This arm is what separates the two: it must pass under
        component validation and would pass under equality as well, whereas
        the next one distinguishes them.
        """

        cfg, _model = _control_with_literal_runner_copy("model")
        _validate(cfg)

    def test_a_runner_may_carry_a_component_the_root_does_not_spell(self) -> None:
        """The divergence nobody cares about, which equality would refuse.

        Adding an inert annotation to the runner's own copy leaves every rule
        satisfied. Whole-config root-equality would reject it, and the failure
        would arrive as a run that cannot start rather than as a red test --
        which is why the ratified contract prefers closing over the components
        actually constructed.
        """

        cfg, model = _control_with_literal_runner_copy("model")
        model.trace_name = "tpen_runner_view"
        _validate(cfg)

    def test_a_frozen_readout_in_the_runner_copy_is_refused(self) -> None:
        cfg, model = _control_with_literal_runner_copy("model")
        model.readout.trainable = False
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.model.readout.trainable" in _paths(caught.value)

    def test_a_global_gradient_clip_in_the_runner_copy_is_refused(self) -> None:
        cfg, trainer = _control_with_literal_runner_copy("trainer")
        trainer.gradient_clip_norm = 1.0
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.trainer.gradient_clip_norm" in _paths(caught.value)

    def test_an_unadmitted_cusp_law_in_the_runner_copy_is_refused(self) -> None:
        """A CONSTRUCTOR-VALID law: it builds fine and trains into a bad tail."""

        cfg, model = _control_with_literal_runner_copy("model")
        model.factors[1].law._target_ = "tpen.nn.CurvatureElectronNucleusCuspLaw"
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.model.factors[1].law._target_" in _paths(caught.value)

    def test_an_omitted_nonfinite_policy_in_the_runner_copy_is_refused(self) -> None:
        """OMISSION, not a bad value: the inherited behaviour is to mask."""

        cfg, trainer = _control_with_literal_runner_copy("trainer")
        del trainer.nonfinite_local_energy_policy
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.trainer.nonfinite_local_energy_policy" in _paths(caught.value)

    def test_an_unadmitted_method_in_the_runner_copy_is_refused(self) -> None:
        cfg, optimizer = _control_with_literal_runner_copy("optimizer")
        optimizer._target_ = "tpen.training.sr.StochasticReconfigurationUpdate"
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.optimizer._target_" in _paths(caught.value)

    def test_a_moved_model_coordinate_in_the_runner_copy_is_refused(self) -> None:
        cfg, model = _control_with_literal_runner_copy("model")
        model.embedding.max_order = 3
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.model.embedding.max_order" in _paths(caught.value)

    def test_a_runner_view_with_no_optimizer_is_not_a_missing_method_violation(self) -> None:
        """The presence-scoping control for the method rule.

        An evaluation runner constructs a model and declares no optimizer.
        Demanding one of every component view would refuse it, and the refusal
        would arrive as a run that cannot start. The ROOT still requires one,
        which the existing ``test_rejects_a_config_with_no_optimizer`` pins.

        THE ``model`` KEY IS LOAD-BEARING AND MEASURED. An earlier version of
        this arm used ``tasks: []``, which carries no component key at all, so
        the runner was never made into a view and the arm passed whether the
        rule was presence-scoped or not. It was VACUOUS: flipping
        ``require_optimizer`` to ``True`` left the whole suite green while
        genuinely changing behaviour. With ``model`` present the runner is a
        view, and that mutation turns this arm red.
        """

        cfg = _config(
            run={"run_id": "x"},
            runner={
                "_target_": "tpen.runner.Evaluate",
                "model": {"_target_": "tpen.nn.TPENWaveFunction", "trace_name": "tpen"},
            },
        )
        _validate(cfg)

    def test_a_component_nested_deeper_than_one_level_is_still_swept(self) -> None:
        """The view is any node carrying a component key, not `runner.*` alone.

        A fixed one-level path list would have passed this while leaving the
        shape it does not name unjudged -- the same failure as reading the root
        only, one level down.
        """

        cfg = _config(
            run={"run_id": "x"},
            runner={
                "_target_": "tpen.runner.Train",
                "stages": {
                    "warmup": {
                        "trainer": {
                            "_target_": "tpen.training.trainer.VMCTrainer",
                            "nonfinite_local_energy_policy": "fail",
                            "gradient_clip_norm": 2.0,
                        }
                    }
                },
            },
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "runner.stages.warmup.trainer.gradient_clip_norm" in _paths(caught.value)


class TestResolverAdmissionIsAnAllowlist:
    """A denylist was beaten TWICE here, which is why the instrument changed.

    First by nesting: a forbidden resolver inside a permitted one was invisible
    to a pattern that stopped at the first closing brace. Then by an unnamed
    sibling: ``oc.decode`` was refused BY NAME for evaluating arbitrary text,
    and ``oc.create`` does the same job under a name nobody had listed, and
    additionally RE-RESOLVES interpolations in its output.

    MEASURED before the change, at head ``38edc53``: a config carrying
    ``oc.create`` around a YAML string whose unicode escape hid the
    interpolation opener from the raw sweep resolved ``run.run_id`` to
    ``rank_7`` from ``SLURM_PROCID``. Every rank would resolve a DIFFERENT run
    identity while the rank-divergence rule passed -- it only asks that
    ``run_id`` be non-null -- and the canonical digest would faithfully report a
    difference it could not explain.

    That is this slice's own class applied to the resolver rule: a rule that
    reads NAMES, defeated by something supplying the same capability without
    that name.
    """

    def test_the_unnamed_sibling_of_a_denied_resolver_is_refused(self) -> None:
        cfg = _config(
            run={"run_id": "x"},
            runtime={"blob": "${oc.create:'a: 1'}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-resolver" in _rules(caught.value)

    def test_the_escape_that_motivated_this_is_refused_end_to_end(self) -> None:
        """The exhibit, spelled the way it was measured.

        The YAML text carries a unicode escape so the raw tree contains no
        literal interpolation opener at all -- which is what made every
        pre-existing raw sweep pass.
        """

        hidden = 'a: "\u0024{oc.env:SLURM_PROCID}"'
        cfg = _config(
            run={"blob": "${oc.create:'" + hidden + "'}", "run_id": "rank_${run.blob.a}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-resolver" in _rules(caught.value)

    def test_a_named_forbidden_resolver_keeps_its_specific_reason(self) -> None:
        """The denylist is retained for MESSAGES and must actually still fire.

        A config reaching for ``oc.env`` should be told that it reads
        process-local state, not merely that some resolver is unadmitted. The
        denylist is therefore checked FIRST -- and this test is what makes that
        claim true rather than a comment, since the allowlist alone would
        refuse the same config with a generic reason.
        """

        cfg = _config(run={"run_id": "x"}, runtime={"seed": "${oc.env:RANK,0}"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        rules = _rules(caught.value)
        assert "forbidden-resolver" in rules
        assert "unadmitted-resolver" not in rules, (
            "one expression must yield one finding; reporting both reads as two defects"
        )


class TestTheReferenceModuleCannotArriveAsDATA:
    """A generic importer carries the module path under a key nobody tokenizes.

    MEASURED at ``38edc53``: ``{_target_: importlib.import_module, name:
    tpen.hi_manifest}`` and ``{_target_: hydra.utils.get_method, path:
    tpen.hi_manifest.load_evaluation_manifest}`` both VALIDATED in a free-form
    runner slot, while the direct spelling in the same slot was refused.

    ENUMERATING THE IMPORTERS WOULD BE THE SAME MISTAKE AGAIN -- ``get_class``,
    ``get_object``, ``pydoc.locate``, ``pkgutil.resolve_name``, and whatever
    lands next. Whatever the importer, the module path must appear SOMEWHERE as
    a string, so the string is what is checked.
    """

    @pytest.mark.parametrize(
        "spec",
        [
            {"_target_": "importlib.import_module", "name": "tpen.hi_manifest"},
            {
                "_target_": "hydra.utils.get_method",
                "path": "tpen.hi_manifest.load_evaluation_manifest",
            },
            {"_target_": "pydoc.locate", "name": "tpen.hi_manifest.reference_energy"},
            # Not an importer at all -- the path as a plain string argument.
            {"_target_": "tpen.callback.Metadata", "note": "tpen.hi_manifest"},
        ],
    )
    def test_the_module_path_is_refused_wherever_it_appears(self, spec: dict) -> None:
        cfg = _config(run={"run_id": "x"}, runner={"_target_": "tpen.runner.Train", "load": spec})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-target:reference-module" in _rules(caught.value)

    def test_the_token_check_stays_scoped_to_target(self) -> None:
        """Only MODULE IDENTITY widened; the token check did not.

        Widening the token check to every value would refuse a run directory
        named ``outputs/baseline``, which executes nothing. Exact module
        identity is safe to widen precisely because it is not a token.
        """

        _validate(_config(run={"root": "outputs/baseline_sweep", "run_id": "x"}))

    def test_the_spellings_this_check_does_NOT_match_are_unimportable(self) -> None:
        """Pins the premise of a deliberate NON-decision.

        Auditing the widened identity check found three spellings it does not
        match. TWO of them -- a bare ``hi_manifest`` and an upper-case
        ``TPEN.HI_MANIFEST`` -- cannot import the module, and this test pins
        that. The third, a filesystem path, CAN, and is filed rather than
        pinned; see the paragraph below.

        MEASURED rather than reasoned: the first two raise ``ModuleNotFoundError``
        -- Python's finder is case-sensitive for module NAMES regardless of the
        filesystem's case sensitivity, which is why the upper-case form fails
        even on macOS.

        **THE PATH FORM IS NOT COVERED BY THIS TEST AND IS NOT UNREACHABLE.**
        The first version of this docstring said reaching it would need
        ``spec_from_file_location`` chained into ``module_from_spec``, which
        needs ``_args_`` and is therefore refused. That was WRONG:
        ``runpy.run_path`` takes the path directly as a keyword and executes the
        module source, and a reviewer measured it validating. The error was
        testing ONE mechanism, ``importlib``, and generalising to a
        class-level negative -- the same census-versus-reachability mistake this
        lane inherited a warning about. The filesystem spelling is tracked as
        its own filed item.

        This test exists because "I checked and there was nothing reachable" is
        a claim that decays silently: a package rename, a new top-level module,
        or a ``sys.path`` change could make one of these importable while every
        other test stays green. Then this one fails and the non-decision gets
        revisited.

        The alternative -- refusing these spellings anyway -- was rejected as
        exactly the error this slice has made three times: naming residuals that
        are real but unreachable, which reads as thoroughness while costing
        real coverage nowhere.
        """

        import importlib

        for unreachable in ("hi_manifest", "TPEN.HI_MANIFEST"):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(unreachable)
        # The control: the spelling the check DOES match is the one that works.
        assert importlib.import_module(REFERENCE_MANIFEST_MODULE) is not None

    @pytest.mark.parametrize(
        ("label", "spec"),
        [
            (
                "relative import splits the path across two arguments",
                {"_target_": "importlib.import_module", "name": ".hi_manifest", "package": "tpen"},
            ),
            (
                "filesystem spelling, which EXECUTES the module source",
                {"_target_": "runpy.run_path", "path_name": "tpen/hi_manifest.py"},
            ),
            (
                # VALIDATION-GAP ARM ONLY: accepted by the schema, but does NOT
                # reach the module through Hydra (DictConfig != dict).
                "import hook with a relative level (does not reach, via Hydra)",
                {
                    "_target_": "builtins.__import__",
                    "name": "hi_manifest",
                    "globals": {"__package__": "tpen"},
                    "level": 1,
                },
            ),
        ],
    )
    def test_KNOWN_OPEN_a_path_never_written_whole_is_not_caught(
        self, label: str, spec: dict
    ) -> None:
        """A RED-BY-DESIGN record of a known-open escape, filed as `fb70cf90`.

        This test asserts the CURRENT, WRONG behaviour on purpose. Every spec
        below VALIDATES, which is the defect under record.

        **THEY DO NOT ALL REACH THE MODULE, and the earlier wording here said
        they did.** Measured through Hydra, which is the path that matters:
        the relative-import and ``runpy.run_path`` specs REACH it; the
        ``builtins.__import__`` spec does NOT -- it dies
        ``TypeError('globals must be a dict')``, because OmegaConf hands Hydra a
        ``DictConfig`` for that argument rather than a real dict. It reaches the
        module when called directly from Python with a real dict, which is how
        the wrong claim was made: **the mechanism was verified in a shell and
        the CONSTRUCTION PATH was not.**

        The third spec is kept, as a validation-gap arm rather than a
        reachability arm, and labelled so.

        Why assert the defect rather than delete the case: a known-open hole
        with no executable trace is indistinguishable from one nobody found. If
        a later slice closes `fb70cf90`, this test goes RED and whoever fixed it
        is pointed at the item and at this docstring, rather than discovering a
        mysterious passing assertion about a config that should be refused.

        THE CLASS IS NOT THESE THREE SPELLINGS. It is "a path that is never
        written whole": split across arguments, or in filesystem form. String
        identity cannot close the filesystem half at all -- absolute paths,
        ``./`` prefixes, symlinks and case-insensitive filesystems all spell the
        same file -- so the remedy must govern the SLOT or the CONSUMER. Do not
        close this by adding three more strings to a list.

        Impact bound, and it should stay bounded: the holder becomes reachable,
        which is the hazard the rule names. The reference NUMBERS live in
        manifest files, and with ``_args_`` refused no config-only shape can
        CALL the loader.
        """

        cfg = OmegaConf.load(_CONTROL_CONFIG)
        OmegaConf.update(cfg, "runner.load", spec, force_add=True)
        _validate(cfg)  # known-open: this SHOULD refuse and does not

        # The control that keeps this honest: the dotted spelling in the very
        # same slot IS refused, so the gap is the spelling and not the slot.
        dotted = OmegaConf.load(_CONTROL_CONFIG)
        OmegaConf.update(
            dotted,
            "runner.load",
            {"_target_": "importlib.import_module", "name": REFERENCE_MANIFEST_MODULE},
            force_add=True,
        )
        with pytest.raises(ClosedSchemaError):
            _validate(dotted)

    def test_a_similarly_named_module_is_not_caught(self) -> None:
        """The identity check must not fire on a prefix that merely looks alike."""

        _validate(
            _config(
                run={"run_id": "x"},
                runner={"_target_": "tpen.runner.Train", "note": "tpen.hi_manifesto.thing"},
            )
        )


class TestExecutableTargets:
    """A ``_target_`` is executable, so its VALUE is checked, not only its key.

    ``ForbiddenSurface.matches`` tokenizes keys. That is right for data: a value
    is inert and the key names what it is. A ``_target_`` inverts it -- the
    value names the code that will run, and the key is always the same word.
    """

    def test_rejects_a_target_naming_the_reference_module(self) -> None:
        cfg = _config(model={"probe": {"_target_": f"{REFERENCE_MANIFEST_MODULE}.reference_energy"}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-target:reference-module" in _rules(caught.value)

    @pytest.mark.parametrize(
        ("target", "rule"),
        [
            ("tpen.diagnostics.energy.ReferenceGapProbe", "forbidden-target:reference"),
            ("tpen.callback.BaselineEnergy", "forbidden-target:reference"),
            ("tpen.callback.AccuracyBandReporter", "forbidden-target:band"),
            ("tpen.training.EarlyStoppingRule", "forbidden-target:stop-rule"),
            ("tpen.training.ContinuationLadder", "forbidden-target:continuation"),
        ],
    )
    def test_rejects_a_target_whose_tokens_name_a_forbidden_family(
        self, target: str, rule: str
    ) -> None:
        """The token rule, on targets OUTSIDE the reference module.

        Every one of these lives somewhere the module rule cannot see, so this
        arm measures the token rule rather than re-measuring the module rule.
        """

        cfg = _config(model={"probe": {"_target_": target}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert rule in _rules(caught.value)

    def test_the_two_rules_do_not_subsume_each_other(self) -> None:
        """Each rule is measured against the case the other misses.

        ``load_evaluation_manifest`` carries no forbidden token, so the token
        rule alone would admit the function that reads the reference file.
        ``ReferenceGapProbe`` is outside the manifest module, so the module rule
        alone would admit it. Neither rule is redundant.
        """

        token_blind = f"{REFERENCE_MANIFEST_MODULE}.load_evaluation_manifest"
        assert not any(
            surface.matches(token_blind) for surface in HI_TRAIN_POLICY.forbidden_surfaces
        ), "the token rule was expected to be blind to this target; if it now sees it, this test no longer measures what it claims"

        module_blind = "tpen.diagnostics.energy.ReferenceGapProbe"
        assert not module_blind.startswith(f"{REFERENCE_MANIFEST_MODULE}.")

        for target in (token_blind, module_blind):
            with pytest.raises(ClosedSchemaError):
                _validate(_config(model={"probe": {"_target_": target}}))

    def test_rejects_a_target_at_any_depth(self) -> None:
        """Depth is not a defence: the sweep walks the whole resolved tree."""

        cfg = _config(
            model={"a": {"b": {"c": {"_target_": f"{REFERENCE_MANIFEST_MODULE}.reference_energy"}}}}
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-target:reference-module" in _rules(caught.value)

    @pytest.mark.parametrize(
        "target",
        [
            # Every one of these is a real target in the shipped control config.
            "tpen.nn.TPENWaveFunction",
            "tpen.nn.initialization.TorchInitializer",
            "tpen.data.atomic_configuration.AtomicConfiguration",
            "tpen.physics.potential.NucleusNucleusPotential",
            "tpen.sampling.metropolis.MetropolisSampler",
            "tpen.equivariance.checks.FullModelEquivarianceChecker",
            "tpen.accelerator.TorchAllocatorPeakProbe",
            "tpen.checkpoint.TrainResume",
            "torch.optim.Adam",
        ],
    )
    def test_accepts_the_targets_the_study_actually_constructs(self, target: str) -> None:
        """The over-restriction control, one arm per target.

        A token set one word too wide would refuse a real component. Read as a
        set they would be one assertion and the failure would name none of
        them; one arm each is what makes a refusal say WHICH target it refused.
        """

        _validate(_config(model={"probe": {"_target_": target}}))

    def test_a_forbidden_token_in_an_ordinary_value_is_still_permitted(self) -> None:
        """The rule is scoped to ``_target_``, not to every string in the tree.

        Widening it to all values would refuse a run directory named
        ``outputs/baseline`` and a docstring-like comment field, neither of
        which executes anything.
        """

        _validate(
            _config(run={"root": "outputs/baseline_sweep", "run_id": "control_0001"})
        )


class TestClosedSections:
    def test_rejects_an_undeclared_top_level_section(self) -> None:
        cfg = _config(diagnostics={"kind": "energy"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unknown-field" in _rules(caught.value)
        assert "diagnostics" in _paths(caught.value)

    def test_accepts_the_declared_sections(self) -> None:
        _validate(
            _config(
                experiment={"name": "hi"},
                run={"root": "outputs", "run_id": "hi_0001"},
                runtime={"seed": 0},
                system={"n_particles": 2},
                model={"channels": 32},
                trainer={"max_steps": 10, "nonfinite_local_energy_policy": "fail"},
            )
        )


class TestForbiddenResolvers:
    """Environment and clock interpolation are how two ranks could diverge."""

    def test_rejects_an_environment_interpolation(self) -> None:
        cfg = _config(runtime={"seed": "${oc.env:RANK,0}"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-resolver" in _rules(caught.value)

    def test_rejects_a_clock_interpolation(self) -> None:
        cfg = _config(run={"root": "outputs/${now:%Y-%m-%d}"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-resolver" in _rules(caught.value)

    def test_accepts_an_ordinary_node_reference(self) -> None:
        _validate(
            _config(system={"spatial_dim": 3}, model={"spatial_dim": "${system.spatial_dim}"})
        )

    @pytest.mark.parametrize(
        "expression",
        [
            "${oc.select:missing,${oc.env:HI_LANE_NUMBER}}",
            "${oc.select:a,${oc.select:b,${oc.env:HI_LANE_NUMBER}}}",
            "${${oc.env:HI_LANE_NUMBER}}",
            "${oc.select:missing,${now:%S}}",
        ],
    )
    def test_rejects_a_forbidden_resolver_nested_inside_a_permitted_one(
        self, expression: str
    ) -> None:
        """The policy-level half of the nested-resolver rule.

        ``oc.select`` is not forbidden and never should be. What is forbidden is
        the ``oc.env`` INSIDE it, which resolves to whatever the launching
        process happens to carry -- so the same file resolves to different
        values on two ranks and produces two canonical train identities. The
        mechanism is exercised in ``tests/unit/test_config_schema.py``; this
        asserts the HI policy actually consumes it.
        """

        cfg = _config(runtime={"seed": expression})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "forbidden-resolver" in _rules(caught.value)

    def test_a_permitted_resolver_is_now_refused_by_the_ALLOWLIST(self) -> None:
        """DELIBERATE WIDENING, and this test was inverted rather than deleted.

        It previously asserted that ``oc.select`` is ACCEPTED, as the
        over-restriction control for the nested-resolver rule under a DENYLIST.
        The policy has since moved to an allowlist, and the allowlist is empty,
        so every resolver CALL is refused and this configuration no longer
        validates.

        Inverted rather than removed on purpose. A deleted test leaves no trace
        that the behaviour changed; this one records that the change was a
        decision, names the rule that now fires, and will fail loudly if
        somebody re-admits ``oc.select`` without saying why.

        The doctrine change is not a preference. A denylist was beaten twice
        here -- once by nesting, once by ``oc.create`` supplying ``oc.decode``'s
        text-evaluation capability under a name nobody had listed.
        """

        cfg = _config(
            system={"spatial_dim": 3},
            model={"spatial_dim": "${oc.select:model.missing,${system.spatial_dim}}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-resolver" in _rules(caught.value)

    def test_plain_node_references_are_untouched_by_the_allowlist(self) -> None:
        """THE over-restriction control, and the one that now carries the weight.

        An empty resolver allowlist would be catastrophic if it also refused
        node references: the shipped config contains 32 of them and would not
        validate. The discriminator is that a reference names CONFIGURATION and
        a call names a RESOLVER, so nesting and computed paths stay writable.
        """

        _validate(
            _config(
                system={"spatial_dim": 3},
                runtime={"seed": 0},
                model={
                    "spatial_dim": "${system.spatial_dim}",
                    "seed": "${runtime.seed}",
                    "nested": "${model.spatial_dim}",
                },
            )
        )

    def test_the_shipped_config_uses_no_resolver_calls(self) -> None:
        """The measurement that makes an EMPTY allowlist defensible.

        Refusing every resolver call is only reasonable while nothing shipped
        makes one. That is asserted here rather than written in a comment,
        because a comment keeps saying so after it stops being true.
        """

        raw = OmegaConf.to_container(OmegaConf.load(_CONTROL_CONFIG), resolve=False)
        calls, references = [], 0
        for path, _key, value in iter_nodes(raw):
            if not isinstance(value, str):
                continue
            for expression in iter_interpolations(value):
                name, is_call = split_resolver(expression)
                if is_call:
                    calls.append((path, name))
                else:
                    references += 1
        assert calls == [], calls
        assert references > 0, "no interpolations found at all; this test would pass vacuously"


class TestAdmittedMethods:
    """An unavailable method must stay visibly unavailable, never become Adam."""

    def test_accepts_adam(self) -> None:
        _validate(_config(optimizer={"_target_": "torch.optim.Adam", "lr": 0.005}))

    def test_rejects_a_method_that_is_not_admitted(self) -> None:
        cfg = _config(optimizer={"_target_": "tpen.training.sr.StochasticReconfiguration"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-method" in _rules(caught.value)

    def test_the_refusal_states_what_admission_requires(self) -> None:
        """A refusal that does not say what would change it is a dead end."""

        cfg = _config(optimizer={"_target_": "somewhere.KFAC"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        detail = " ".join(r.detail for r in caught.value.rejections)
        assert "kfac=unavailable" in detail
        assert "compatibility gate" in detail

    def test_every_unavailable_method_appears_in_the_roster(self) -> None:
        """The four deferred methods are tracked, not merely absent."""

        by_name = {entry.method: entry for entry in HI_METHOD_ROSTER}
        assert set(by_name) == {"adam", "sr", "kfac", "spring", "linear_method"}
        assert by_name["adam"].admitted
        for name in ("sr", "kfac", "spring", "linear_method"):
            assert not by_name[name].admitted
            assert by_name[name].requires

    def test_a_landed_implementation_is_not_described_as_pending(self) -> None:
        """SR's stated reason must track the repository, which moves under it.

        THE TWO ARMS VARY INDEPENDENTLY, which is the point. One arm is
        REPOSITORY state -- does ``tpen/training/sr.py`` exist -- and moves when
        someone else merges to dev. The other is the roster TEXT and moves only
        when this file is edited. A test that read the reason and checked it
        said "excluded" would be driven from one source and could never fire.

        The failure being guarded is specific and already happened once: this
        entry said SR was waiting for Lane N, Lane N merged, and the entry kept
        saying it. A reader would then satisfy an already-satisfied prerequisite
        and conclude the refusal was a bug. A refusal that outlives its stated
        reason misdirects harder than one with no reason at all.
        """

        from pathlib import Path

        sr_implementation = Path("tpen/training/sr.py")
        entry = next(e for e in HI_METHOD_ROSTER if e.method == "sr")

        if sr_implementation.exists():
            assert "exclud" in entry.requires.lower(), (
                "tpen/training/sr.py exists, so SR is implemented; its roster reason must "
                f"say it is EXCLUDED rather than pending. Got: {entry.requires!r}"
            )

    # A first draft of the check above also blacklisted phrases like
    # "pending work". It was DELETED rather than fixed, and the reason is worth
    # more than the check was: the corrected reason contains the sentence "This
    # is NOT pending work", so a substring scan flagged it. A substring cannot
    # distinguish a claim from its negation -- which is precisely the trap that
    # made the firewall itself match on whole tokens rather than substrings, and
    # it was walked straight back into here.
    #
    # The blacklist was also redundant: requiring an explicit "excluded" is the
    # actual discriminator, and it is the half whose two arms vary
    # independently. A second, cruder instrument pointed at the same question
    # added no coverage and one false failure.

    def test_the_sr_implementation_really_is_present(self) -> None:
        """Positive control for the test above, which is conditional on it.

        Without this the guard would silently no-op if the path were ever
        renamed, and a conditional test that never enters its branch is
        indistinguishable from one that passes.
        """

        from pathlib import Path

        assert Path("tpen/training/sr.py").exists(), (
            "tpen/training/sr.py is gone; the staleness guard above is now inert and "
            "its path needs updating to wherever SR landed"
        )

    def test_sr_remains_unadmitted_whatever_its_reason_says(self) -> None:
        """The refusal and its explanation are separate facts; pin both.

        A correct reason attached to a broken refusal would be worse than the
        stale reason this replaced.
        """

        entry = next(e for e in HI_METHOD_ROSTER if e.method == "sr")
        assert not entry.admitted
        assert entry.target is None
        assert "torch.optim.Adam" in ADMITTED_METHOD_TARGETS
        assert not any(
            "sr" == e.method and e.admitted for e in HI_METHOD_ROSTER
        )

    def test_an_unavailable_method_declares_no_target(self) -> None:
        """A module path for an unimplemented method would be a false claim."""

        for entry in HI_METHOD_ROSTER:
            if not entry.admitted:
                assert entry.target is None

    def test_rejects_a_config_with_no_optimizer(self) -> None:
        cfg = OmegaConf.create({"schema": HI_TRAIN_SCHEMA, "trainer": {"max_steps": 1}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "missing-method" in _rules(caught.value)

    def test_rejects_a_changed_adam_beta1(self) -> None:
        """beta2 is the scanned moment; beta1 is fixed for every cell."""

        cfg = _config(optimizer={"_target_": "torch.optim.Adam", "betas": [0.5, 0.999]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "frozen-coordinate" in _rules(caught.value)
        assert "optimizer.betas[0]" in _paths(caught.value)

    def test_accepts_both_scanned_beta2_levels(self) -> None:
        for beta2 in (0.99, 0.999):
            _validate(_config(optimizer={"_target_": "torch.optim.Adam", "betas": [0.9, beta2]}))

    def test_accepts_every_scanned_learning_rate(self) -> None:
        """lr is a scan coordinate with four levels and must not be pinned."""

        for lr in (0.0005, 0.0015, 0.005, 0.015):
            _validate(_config(optimizer={"_target_": "torch.optim.Adam", "lr": lr}))

    def test_rejects_weight_decay(self) -> None:
        cfg = _config(optimizer={"_target_": "torch.optim.Adam", "weight_decay": 0.01})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "optimizer.weight_decay" in _paths(caught.value)

    def test_an_omitted_fixed_coordinate_is_not_a_violation(self) -> None:
        """The fixed values ARE the library defaults; omitting them is normal.

        Requiring them to be spelled out would reject every configuration that
        simply does not mention eps -- including the compliant one.
        """

        _validate(_config(optimizer={"_target_": "torch.optim.Adam", "lr": 0.005}))


class TestFrozenArchitecture:
    """Coordinates no arm may move -- and, just as importantly, ones every arm may."""

    @pytest.mark.parametrize(
        ("section", "body"),
        [
            ("system", {"spatial_dim": 2}),
            ("runtime", {"dtype": "float32"}),
            # Carries a ``_target_`` because that is what identifies a TERM.
            # See TestHamiltonianTermsAreFoundByShapeNotByKey for why the rule
            # matches on the target rather than on the key, and
            # test_a_term_without_a_target_is_not_constructible_anyway for why
            # narrowing to targets loses no real coverage.
            (
                "hamiltonian_terms",
                {
                    "electron_nucleus": {
                        "_target_": "tpen.physics.potential.ElectronNucleusPotential",
                        "eps": 1e-8,
                    }
                },
            ),
        ],
    )
    def test_rejects_a_moved_scalar(self, section: str, body: dict) -> None:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(_config(**{section: body}))
        assert "frozen-coordinate" in _rules(caught.value)

    def test_accepts_the_frozen_scalar_values(self) -> None:
        _validate(
            _config(
                system={"spatial_dim": 3},
                runtime={"dtype": "float64"},
                hamiltonian_terms={
                    "electron_nucleus": {
                        "_target_": "tpen.physics.potential.ElectronNucleusPotential",
                        "eps": 0.0,
                    }
                },
            )
        )

    @pytest.mark.parametrize(
        ("key", "bad"),
        [("max_order", 3), ("max_virtual_order", 1), ("implementation", "slow")],
    )
    def test_rejects_a_moved_model_coordinate_at_any_depth(self, key: str, bad: object) -> None:
        """Nesting varies with the producer policy, so the rule is depth-free.

        A1/A2 swap tensor, linear and hybrid producers, which changes where
        these keys sit. A fixed path would stop matching on some arms and pass
        them by default.
        """

        cfg = _config(model={"layers": [{"mixing": {key: bad}}]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "frozen-coordinate" in _rules(caught.value)
        assert f"model.layers[0].mixing.{key}" in _paths(caught.value)

    def test_rejects_v1_virtual_support(self) -> None:
        """A3 is fixed at 2; there is no V1 arm."""

        cfg = _config(model={"layers": [{"mixing": {"max_virtual_order": 1}}]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "frozen-coordinate" in _rules(caught.value)

    def test_rejects_a_global_gradient_clip(self) -> None:
        cfg = _config(trainer={"gradient_clip_norm": 1.0})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "trainer.gradient_clip_norm" in _paths(caught.value)

    def test_a_null_gradient_clip_is_accepted(self) -> None:
        """Null is how a config says "no clipping" explicitly."""

        _validate(
            _config(
                trainer={
                    "gradient_clip_norm": None,
                    "max_steps": 10,
                    "nonfinite_local_energy_policy": "fail",
                }
            )
        )

    @pytest.mark.parametrize(
        ("section", "body"),
        [
            # A4/A5 channels, A6 activations, A7 embedding width/depth,
            # A8 update rule, A9 producer init. None may be pinned.
            ("model", {"layers": [{"mixing": {"channels": 48}}]}),
            ("model", {"layers": [{"mixing": {"activation": {"_target_": "torch.nn.Tanh"}}}]}),
            ("model", {"embedding": {"hidden_channels": 256, "num_hidden_layers": 2}}),
            ("model", {"layers": [{"update": {"_target_": "tpen.nn.ReplaceUpdater"}}]}),
            ("model", {"layers": [{"mixing": {"initial_weight": 1.0}}]}),
        ],
    )
    def test_does_not_pin_a_scanned_coordinate(self, section: str, body: dict) -> None:
        """Pinning any of these would make the study's own grid unrunnable.

        This is the half of the check that a "reject more" instinct gets wrong.
        The scan varies producer policy, channels, activations, embedding
        width/depth, the feature update rule and five initializations; a schema
        that froze them would reject the arms it exists to serve.
        """

        _validate(_config(**{section: body}))


class TestDeclaredTrainability:
    """Trainability must be declared, never inherited."""

    def test_rejects_a_readout_that_omits_trainable(self) -> None:
        """The exact he-v1 defect: silent, total, and 300,000 updates long.

        PfaffianReadout defaults trainable=False, and under that default the
        channel weights appear in neither named_parameters() nor state_dict().
        Nothing logs them and no gradient touches them, so the only way to
        notice is to require the declaration.
        """

        cfg = _config(model={"readout": {"_target_": "tpen.nn.readout.PfaffianReadout", "channels": 32}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-trainability" in _rules(caught.value)
        assert "model.readout.trainable" in _paths(caught.value)

    def test_rejects_a_readout_declared_untrainable(self) -> None:
        cfg = _config(model={"readout": {"channels": 32, "trainable": False}})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-trainability" in _rules(caught.value)

    def test_accepts_a_readout_declared_trainable(self) -> None:
        _validate(_config(model={"readout": {"channels": 32, "trainable": True}}))

    def test_a_config_with_no_readout_is_not_a_trainability_violation(self) -> None:
        """An incomplete config is a different defect from a frozen parameter."""

        _validate(_config(model={"embedding": {"out_channels": 32}}))

    def test_rejects_an_ee_cusp_that_omits_trainable_range(self) -> None:
        cfg = _config(model={"factors": [{"_target_": "tpen.nn.ElectronElectronCusp"}]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.factors[0].trainable_range" in _paths(caught.value)

    def test_rejects_an_en_cusp_whose_law_omits_trainable(self) -> None:
        cfg = _config(
            model={
                "factors": [
                    {
                        "_target_": "tpen.nn.ElectronNucleusCusp",
                        "law": {"_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw"},
                    }
                ]
            }
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.factors[0].law.trainable" in _paths(caught.value)

    def test_accepts_fully_declared_factors(self) -> None:
        """Uses the ADMITTED law.

        This test previously named ``CurvatureElectronNucleusCuspLaw``, which
        the law allowlist now refuses for this study. The change is deliberate,
        not incidental: declaring trainability correctly is no longer
        sufficient, because a fully declared unconstrained-tail law is exactly
        the configuration the allowlist exists to stop.
        """

        _validate(
            _config(
                model={
                    "factors": [
                        {"_target_": "tpen.nn.ElectronElectronCusp", "trainable_range": True},
                        {
                            "_target_": "tpen.nn.ElectronNucleusCusp",
                            "law": {
                                "_target_": "tpen.nn.TailSafeElectronNucleusCuspLaw",
                                "trainable": True,
                            },
                        },
                        {
                            "_target_": "tpen.nn.BoundedTwoCoefficientJastrow",
                            "trainable": True,
                        },
                    ]
                }
            )
        )

    def test_an_unknown_factor_is_not_given_a_trainability_rule(self) -> None:
        """A factor with no registered rule is not guessed at.

        ``BoundedTwoCoefficientJastrow`` now HAS a rule, registered when the
        factor landed rather than when a config first used it. This test keeps
        using a genuinely unregistered name, so it still measures the absence
        of guessing rather than the absence of that one entry.
        """

        _validate(_config(model={"factors": [{"_target_": "tpen.nn.SomeFutureJastrow"}]}))

    def test_rejects_a_jastrow_that_omits_trainable(self) -> None:
        """Both coefficients start at zero, so an inherited false is invisible.

        The factor would evaluate to exactly 0 for the whole run -- an identity
        multiplier -- with nothing in named_parameters(), nothing in
        state_dict(), and no log line saying so.
        """

        cfg = _config(model={"factors": [{"_target_": "tpen.nn.BoundedTwoCoefficientJastrow"}]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.factors[0].trainable" in _paths(caught.value)


class TestTheElectronNucleusLawIsAdmitted:
    """Selecting the wrong cusp law must fail closed, not run unbounded.

    Independent of the trainability rule above. That rule asks whether the law
    is TRAINED; this asks whether it is the RIGHT LAW, and a trainable
    unconstrained-tail law satisfies the first completely while still being
    able to train its way into a non-normalizable tail.
    """

    def _factors(self, law: object) -> dict:
        return {
            "factors": [
                {"_target_": "tpen.nn.ElectronNucleusCusp", "law": law},
            ]
        }

    def test_rejects_the_unconstrained_tail_law(self) -> None:
        """The predecessor is refused even when fully and correctly declared."""

        cfg = _config(
            model=self._factors(
                {"_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw", "trainable": True}
            )
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-cusp-law" in _rules(caught.value)
        assert "model.factors[0].law._target_" in _paths(caught.value)

    def test_accepts_the_tail_safe_law(self) -> None:
        """Over-restriction control.

        Without this, an allowlist that admitted NOTHING would satisfy every
        rejection test above and would only be discovered when a real run
        could not start.
        """

        _validate(
            _config(
                model=self._factors(
                    {"_target_": "tpen.nn.TailSafeElectronNucleusCuspLaw", "trainable": True}
                )
            )
        )

    def test_rejects_a_law_that_declares_no_target(self) -> None:
        cfg = _config(model=self._factors({"trainable": True}))
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-cusp-law" in _rules(caught.value)

    def test_an_absent_law_is_reported_once_as_a_trainability_defect(self) -> None:
        """One omission must not produce two differently named rejections.

        A missing law already fails the trainability rule, which requires
        ``law.trainable`` to be declared true. Reporting it a second time as an
        unadmitted law would read as two independent defects and send a reader
        looking for a second fix that does not exist.
        """

        cfg = _config(model={"factors": [{"_target_": "tpen.nn.ElectronNucleusCusp"}]})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "model.factors[0].law.trainable" in _paths(caught.value)
        assert "unadmitted-cusp-law" not in _rules(caught.value)

    def test_the_nonfinite_policy_must_be_declared(self) -> None:
        """Absence is a refusal, because the inherited value is a biased estimator.

        Masking non-finite local-energy rows drops a systematically selected
        subsample -- those rows occur where the local energy is pathological --
        so a run that inherits it is reporting under a known-biased estimator
        without saying so. There is deliberately no default here.
        """

        cfg = _config(trainer={"max_steps": 10})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-nonfinite-policy" in _rules(caught.value)
        assert "trainer.nonfinite_local_energy_policy" in _paths(caught.value)

    @pytest.mark.parametrize("policy", ["fail", "mask"])
    def test_both_admitted_policies_are_accepted(self, policy: str) -> None:
        """Over-restriction control, and it carries the design intent.

        ``mask`` stays REACHABLE. The point of the rule is not to forbid the
        biased estimator, it is to stop anyone reaching it by omission -- so a
        config that declares it must validate.
        """

        _validate(_config(trainer={"nonfinite_local_energy_policy": policy}))

    @pytest.mark.parametrize("policy", ["FAIL", "drop", "true", 1])
    def test_an_unrecognised_policy_is_refused(self, policy) -> None:
        cfg = _config(trainer={"nonfinite_local_energy_policy": policy})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "undeclared-nonfinite-policy" in _rules(caught.value)

    def test_a_factor_that_is_not_the_en_cusp_is_left_alone(self) -> None:
        """The rule is scoped to the electron-nucleus cusp, not to any 'law' key."""

        _validate(
            _config(
                model={
                    "factors": [
                        {
                            "_target_": "tpen.nn.SomeFutureFactor",
                            "law": {"_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw"},
                        }
                    ]
                }
            )
        )


class TestRankInvariance:
    """The resolved configuration must be identical in every process."""

    def test_rejects_a_null_run_id(self) -> None:
        """MEASURED: generate_run_id ends in uuid4().hex[:6].

        prepare_run_context fills a null run_id from generate_run_id, whose
        suffix is RANDOM. So this is not a clock-skew risk that might not bite
        -- every process computes a different identifier, always, and each rank
        would write to its own run directory.
        """

        cfg = _config(run={"root": "outputs", "run_id": None})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "rank-divergent-field" in _rules(caught.value)
        assert "run.run_id" in _paths(caught.value)

    def test_accepts_an_explicit_run_id(self) -> None:
        _validate(_config(run={"root": "outputs", "run_id": "hi_o1_0007"}))

    def test_a_config_without_a_run_section_is_not_judged(self) -> None:
        """Absence of the section is incompleteness, not rank divergence."""

        _validate(_config(runtime={"seed": 0}))

    def test_the_random_suffix_really_does_differ_per_call(self) -> None:
        """Positive control for the rule's premise, measured not assumed.

        If generate_run_id were deterministic the rule above would be pinning a
        hazard that does not exist. This asserts the premise directly, so the
        rule cannot outlive its own justification silently.
        """

        from tpen.artifacts import generate_run_id

        assert generate_run_id("hi") != generate_run_id("hi")

    def test_the_same_config_yields_the_same_identity_under_different_environments(self) -> None:
        """Two ranks, two environments, one canonical identity.

        This is the acceptance contract's "canonical resolved input is
        identical on all ranks", exercised the way it fails: the two processes
        differ exactly in the variables a launcher sets.
        """

        rank_zero = {"RANK": "0", "LOCAL_RANK": "0", "SLURM_PROCID": "0", "HOSTNAME": "holy01"}
        rank_three = {"RANK": "3", "LOCAL_RANK": "1", "SLURM_PROCID": "3", "HOSTNAME": "holy02"}

        first = _config(run={"root": "outputs", "run_id": "hi_o1_0007"}, runtime={"seed": 0})
        second = _config(run={"root": "outputs", "run_id": "hi_o1_0007"}, runtime={"seed": 0})

        _validate(first, env=rank_zero)
        _validate(second, env=rank_three)
        assert canonical_train_identity(first) == canonical_train_identity(second)

    def test_the_identity_does_change_when_the_science_changes(self) -> None:
        """Positive control: an identity that never changes proves nothing.

        Without this, the agreement test above would also pass for a digest
        that returned a constant.
        """

        base = _config(runtime={"seed": 0})
        altered = _config(runtime={"seed": 1})
        assert canonical_train_identity(base) != canonical_train_identity(altered)

    def test_the_identity_ignores_key_order(self) -> None:
        """Mapping iteration order is not part of a configuration's meaning."""

        first = OmegaConf.create({"a": 1, "b": {"x": 1, "y": 2}})
        second = OmegaConf.create({"b": {"y": 2, "x": 1}, "a": 1})
        assert canonical_train_identity(first) == canonical_train_identity(second)


class TestLaunchEnvironment:
    """The firewall names five surfaces; the launch environment is one of them.

    A config-only check would leave a reference reachable through the
    environment of the training process, which the reference-energy firewall
    forbids explicitly and by the same rule that forbids "apparently unused
    fields" in the config.
    """

    def test_rejects_a_reference_bearing_variable(self) -> None:
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(_config(), env={"TPEN_REFERENCE_ENERGY": "-2.903724377034119598"})
        assert "forbidden-environment:reference" in _rules(caught.value)
        assert "TPEN_REFERENCE_ENERGY" in _paths(caught.value)

    def test_rejects_a_variable_that_is_never_read_by_the_config(self) -> None:
        """An unread variable is still a forbidden field.

        This is the case a "does the config use it?" check would miss, and it
        is the one the firewall's "apparently unused fields" clause is about.
        """

        cfg = _config(runtime={"seed": 0})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg, env={"HE_BASELINE_ENERGY": "-2.9"})
        assert "HE_BASELINE_ENERGY" in _paths(caught.value)

    def test_accepts_an_ordinary_environment(self) -> None:
        _validate(
            _config(),
            env={"PATH": "/usr/bin", "SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "0"},
        )

    def test_matches_variable_names_not_values(self) -> None:
        """A value near the reference is not itself a violation.

        Only a name says what a variable means. Matching values would reject
        any variable that happened to hold a similar number -- including a
        legitimate learning rate or tolerance.
        """

        _validate(_config(), env={"SOME_SCALE": "-2.903724377034119598"})

    def test_a_rank_variable_is_not_forbidden(self) -> None:
        """DDP launchers set these; the schema must not fight the launcher.

        Rank facts are forbidden from entering the SCHEMA, which the
        forbidden-resolver check enforces. Their mere presence in the
        environment is normal and is how a launcher communicates topology.
        """

        _validate(_config(), env={"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0"})


class TestAdmittedCallbacks:
    def test_rejects_a_callback_outside_the_admitted_set(self) -> None:
        cfg = _config(callbacks=[{"_target_": "tpen.diagnostics.energy.EnergyDiagnostic"}])
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unadmitted-callback" in _rules(caught.value)

    def test_accepts_every_admitted_callback(self) -> None:
        cfg = _config(callbacks=[{"_target_": name} for name in sorted(ADMITTED_CALLBACK_TARGETS)])
        _validate(cfg)

    def test_a_nested_target_is_not_judged_as_a_callback(self) -> None:
        """A schedule or payload is a constructor argument, not a callback.

        ``tpen.checkpoint.EveryNUpdates`` is not in the callback allowlist and
        must not be, so treating every nested ``_target_`` as a callback
        identity would reject a standard checkpoint block.
        """

        cfg = _config(
            callbacks=[
                {
                    "_target_": "tpen.callback.Checkpoint",
                    "schedule": {"_target_": "tpen.checkpoint.EveryNUpdates", "every_n": 1000},
                }
            ]
        )
        _validate(cfg)

    def test_no_admitted_callback_carries_a_reference_in_its_name(self) -> None:
        """A cheap standing guard on the allowlist itself.

        The allowlist is hand-maintained, so the failure mode is someone adding
        a reference-bearing callback to it. This cannot catch a reference hidden
        behind an innocent class name, and is not claimed to -- it catches the
        careless case for free.
        """

        from tpen.config_schema import tokens_of

        for target in ADMITTED_CALLBACK_TARGETS:
            assert "reference" not in tokens_of(target.rsplit(".", 1)[-1])


class TestUnresolvableConfig:
    def test_a_broken_interpolation_is_reported_as_a_rejection(self) -> None:
        """Preconstruction failure stays one exception type for the caller."""

        cfg = _config(model={"channels": "${missing.node}"})
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert "unresolvable" in _rules(caught.value)

    def test_a_broken_interpolation_does_not_suppress_the_raw_findings(self) -> None:
        """The reference must survive a config that also fails to resolve.

        This is the case where the finding and the thing that hides it share a
        failure domain. If the raw sweep ran only after a successful
        resolution, one unrelated typo would turn a reference-bearing config
        into a bare "does not resolve" -- and the author would fix the typo,
        rerun, and only then discover the reference. On this project the rerun
        can be a cluster job.
        """

        cfg = _config(
            system={"reference_energy": -2.9},
            model={"channels": "${missing.node}"},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert {"unresolvable", "forbidden-surface:reference"} <= _rules(caught.value)
        assert "system.reference_energy" in _paths(caught.value)


class TestEveryFindingIsReported:
    def test_multiple_violations_arrive_together(self) -> None:
        """One cluster cycle per violation is the cost of failing on the first."""

        cfg = _config(
            sneaky={},
            system={"reference_energy": -2.9},
            trainer={"accuracy_band": 1},
        )
        with pytest.raises(ClosedSchemaError) as caught:
            _validate(cfg)
        assert {
            "unknown-field",
            "forbidden-surface:reference",
            "forbidden-surface:band",
        } <= _rules(caught.value)
