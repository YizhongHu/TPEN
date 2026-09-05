"""Tests for the closed helium-importance train schema policy.

The falsifier named in L1a's acceptance contract is "a nested forbidden
reference reaches construction". These tests pin the rejection half of it; the
half that proves nothing was constructed lives with the ``tpen.run`` wiring,
because only there is there anything to construct.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tpen.config_schema import ClosedSchemaError
from tpen.hi_schema import (
    ADMITTED_CALLBACK_TARGETS,
    HI_EXPERIMENT_NAME,
    HI_METHOD_ROSTER,
    HI_TRAIN_SCHEMA,
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
                trainer={"max_steps": 10},
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
            ("hamiltonian_terms", {"electron_nucleus": {"eps": 1e-8}}),
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
                hamiltonian_terms={"electron_nucleus": {"eps": 0.0}},
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

        _validate(_config(trainer={"gradient_clip_norm": None, "max_steps": 10}))

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
        _validate(
            _config(
                model={
                    "factors": [
                        {"_target_": "tpen.nn.ElectronElectronCusp", "trainable_range": True},
                        {
                            "_target_": "tpen.nn.ElectronNucleusCusp",
                            "law": {"_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw", "trainable": True},
                        },
                    ]
                }
            )
        )

    def test_an_unknown_factor_is_not_given_a_trainability_rule(self) -> None:
        """CD4 adds a new factor; it gets its own rule then, not a guessed one."""

        _validate(_config(model={"factors": [{"_target_": "tpen.nn.SomeFutureJastrow"}]}))


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
