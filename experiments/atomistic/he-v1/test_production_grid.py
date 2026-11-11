"""Contracts over the frozen He-v1 production grid (H-F1 predeclaration).

These tests exist because the predeclaration's whole value is that a later
reader can CHECK it. A grid config that merely parses proves nothing: the
failure this slice is guarding against is a bound that was intended and never
declared, which renders `absent` with its observed value retained and reads as
a complete receipt.

Nothing here imports ``tpen`` (``experiments/README.md``); the assertions that
need torch live in ``tests/unit/experiments/test_he_v1_config.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parent


def _load_study_module(name: str) -> ModuleType:
    """Load one study module by path under a study-unique module name.

    A bare ``import collect`` is NOT safe here. Four ``collect.py`` and three
    ``plan.py`` modules exist under ``experiments/``, each study inserting its
    own directory at ``sys.path[0]``, so in a full-suite run the bare name
    resolves to whichever study was imported first -- and the resulting
    ``AttributeError`` looks like a broken function rather than a wrong module.
    Registering the module in ``sys.modules`` BEFORE executing it is also
    required: exec'ing an unregistered module breaks ``@dataclass``, which
    fails inside the standard library and reads as an interpreter fault.
    """

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_torch_side_config_test() -> ModuleType:
    """Load the torch-side He config test so its own stripper can be compared.

    Skips when torch is unavailable, which is the local case: this repo has no
    x86_64 macOS torch wheel, so the comparison runs on Cannon. Skipping is
    honest here -- the assertion is about two implementations agreeing, and it
    cannot be evaluated at all without the module that holds the second one.
    """

    pytest.importorskip("torch", reason="torch-side config test cannot import without torch")
    path = STUDY_DIR.parents[2] / "tests" / "unit" / "experiments" / "test_he_v1_config.py"
    spec = importlib.util.spec_from_file_location("he_v1_torch_side_config_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_study_module("collect")
# Taken from the subject so there is exactly one `gates` module in play: a
# second import would be a different object whose tables could disagree.
gates = collect.gates
plan = collect.plan_stage

GRID = STUDY_DIR / "configs" / "production_grid.yaml"
EVAL = STUDY_DIR / "configs" / "eval.yaml"
TRAIN = STUDY_DIR / "configs" / "train.yaml"

#: Namespace under which singlet purity is interpretable. Under
#: ``full_model_antisymmetry`` the triplet fraction is identically 1.0 by
#: construction, so a bound there gates a constant.
PURITY_NAMESPACE = "eval/spatial_exchange_symmetry"
FORBIDDEN_PURITY_NAMESPACE = "eval/full_model_antisymmetry"


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def grid() -> dict:
    return plan.load_grid_config(GRID)


@pytest.fixture(scope="module")
def thresholds(grid: dict) -> dict:
    values, _ = collect.split_gate_spec(grid["gate_spec"])
    return values


@pytest.fixture(scope="module")
def bindings(grid: dict) -> dict:
    _, values = collect.split_gate_spec(grid["gate_spec"])
    return values


@pytest.fixture(scope="module")
def radial_generator() -> dict:
    return _yaml(EVAL)["evaluation_tasks"]["he_radial_profiles"]["generator"]


class TestGateSpecCompleteness:
    """A forgotten threshold key is the one failure that fails quietly."""

    def test_spec_declares_every_recognized_key(self, thresholds: dict) -> None:
        # The asymmetry this guards: a MISTYPED key raises, but a FORGOTTEN key
        # goes `absent` with its value retained, so a gate someone intended to
        # enforce silently never enforces.
        assert gates.undeclared_spec_keys(thresholds) == ()

    def test_no_key_is_deliberately_omitted(self, thresholds: dict) -> None:
        # H-F1 declares zero omissions. If a future slice needs one it must
        # name it here with a reason rather than dropping it from the spec.
        declared_omissions: tuple[str, ...] = ()
        assert declared_omissions == ()
        assert set(thresholds) == set(gates.ATOM_GATE_SPEC_KEYS)

    @pytest.mark.parametrize("dropped", gates.ATOM_GATE_SPEC_KEYS)
    def test_the_completeness_check_actually_detects_an_omission(
        self, thresholds: dict, dropped: str
    ) -> None:
        # Without this, the completeness assertion above cannot tell a working
        # `undeclared_spec_keys` from one stubbed to return (): the real spec is
        # complete, so the correct answer and the stub's answer coincide. Found
        # by mutation -- stubbing the function left the suite green. An
        # assertion that can only pass is as dangerous as one that can only
        # fail.
        incomplete = {key: value for key, value in thresholds.items() if key != dropped}
        assert gates.undeclared_spec_keys(incomplete) == (dropped,)

    def test_spec_carries_no_unknown_key(self, thresholds: dict) -> None:
        # Unknown keys raise from `evaluate_atom_gates`; assert the spec is
        # already clean so the raise is never how we find out.
        assert sorted(set(thresholds) - set(gates.ATOM_GATE_SPEC_KEYS)) == []
        gates.evaluate_atom_gates({}, spec=thresholds)


class TestNoSecondImplementationOfSpecStripping:
    """Pin the torch-side test's own stripper to the real `split_gate_spec`.

    `tests/unit/experiments/test_he_v1_config.py` cannot import the study
    modules -- `collect.py` reaches its siblings by bare import, so a
    path-based loader cannot fix what the loaded module imports -- and it
    therefore strips the reserved binding block itself. That is the right
    call, and it creates a SECOND IMPLEMENTATION of production's stripping
    logic living in a test.

    Left unpinned it drifts silently: if the reserved-block semantics change,
    that test keeps passing against a spec PRODUCTION WOULD NEVER PRODUCE, and
    every assertion built on it validates the wrong object while looking fully
    cited. Same class as attaching H-C1's memory figure to a model whose
    `hidden_channels` was never set.

    So the two are compared directly, on the SHIPPED grid, here in the suite
    where the real function imports safely.
    """

    def test_test_side_stripper_matches_the_real_split_gate_spec(self) -> None:
        torch_side = _load_torch_side_config_test()
        expected, _ = collect.split_gate_spec(plan.load_grid_config(GRID)["gate_spec"])
        assert torch_side._grid_thresholds() == expected

    def test_the_reserved_key_literal_agrees(self) -> None:
        torch_side = _load_torch_side_config_test()
        assert torch_side._METRIC_NAMESPACE_SPEC_KEY == collect.METRIC_NAMESPACE_SPEC_KEY


class TestSingletPurityNamespaceBinding:
    """Purity must decide on the spatial-exchange number and no other."""

    PURITY_METRICS = (
        "triplet_fraction_mean_under_psi_orig_sq",
        "triplet_fraction_max_under_psi_orig_sq",
        "triplet_fraction_finite_sample_count",
    )

    def test_every_purity_metric_is_bound_to_spatial_exchange(self, bindings: dict) -> None:
        for metric in self.PURITY_METRICS:
            assert bindings.get(metric) == PURITY_NAMESPACE

    def test_no_purity_metric_is_bound_to_full_model_antisymmetry(self, bindings: dict) -> None:
        # Under full label exchange Psi -> -Psi, so u = 0, the sign ratio is -1
        # and f = 1 EXACTLY. A gate bound here would pass or fail on a constant.
        for metric, namespace in bindings.items():
            assert namespace != FORBIDDEN_PURITY_NAMESPACE, metric

    def test_the_constant_one_would_fail_the_declared_purity_bound(
        self, thresholds: dict
    ) -> None:
        # The bound is not vacuous: applied to full_model_antisymmetry's
        # by-construction 1.0 it FAILS, which is why binding the namespace is
        # what makes the gate mean something rather than a formality.
        outcomes = {
            outcome.name: outcome
            for outcome in gates.evaluate_atom_gates(
                {
                    "triplet_fraction_mean_under_psi_orig_sq": 1.0,
                    "triplet_fraction_max_under_psi_orig_sq": 1.0,
                },
                spec=thresholds,
            )
        }
        assert outcomes["triplet_fraction_mean_at_most"].status == "fail"
        assert outcomes["triplet_fraction_max_at_most"].status == "fail"

    def test_production_purity_gates_are_bound_AND_armed_not_absent(
        self, thresholds: dict, bindings: dict
    ) -> None:
        """The shipped config must leave purity ENFORCING, never silently absent.

        This is the test that closes the hazard the collector's leniency
        creates. An auto-requested gate metric that collides and is UNBOUND
        resolves to absent so an unrequested collision cannot fail a row -- but
        the purity metrics collide with `full_model_antisymmetry` BY
        CONSTRUCTION, so if the production grid ever loses its
        `metric_namespaces` binding all three purity gates become no-ops: every
        row passes, the CSV lists every outcome, and the physics check does not
        happen. "Gate absent" is not "gate failed", so nothing would look wrong.

        Asserted against the SHIPPED grid, not a fixture, so it fails when
        someone edits the config rather than when someone edits a helper.
        """

        for metric, threshold_key in (
            ("triplet_fraction_mean_under_psi_orig_sq", "triplet_fraction_mean_max"),
            ("triplet_fraction_max_under_psi_orig_sq", "triplet_fraction_max_max"),
            (
                "triplet_fraction_finite_sample_count",
                "triplet_fraction_finite_sample_count_min",
            ),
        ):
            assert bindings.get(metric) == PURITY_NAMESPACE, (
                f"{metric} is not bound; it collides with full_model_antisymmetry "
                "by construction, so an unbound purity gate is a silent no-op"
            )
            assert thresholds.get(threshold_key) is not None, (
                f"{threshold_key} is undeclared, so its gate is not armed"
            )
            # Armed means the collector treats an unresolved collision on it as a
            # ROW FAILURE rather than tolerating it.
            assert metric in gates.enforcing_metrics(thresholds)

    def test_purity_center_is_analytic_zero(self, thresholds: dict) -> None:
        # A true spatial singlet has triplet fraction 0 exactly, so a perfectly
        # pure row must pass. A bound that a perfect value cannot satisfy would
        # be measuring something other than purity.
        outcomes = {
            outcome.name: outcome
            for outcome in gates.evaluate_atom_gates(
                {
                    "triplet_fraction_mean_under_psi_orig_sq": 0.0,
                    "triplet_fraction_max_under_psi_orig_sq": 0.0,
                },
                spec=thresholds,
            )
        }
        assert outcomes["triplet_fraction_mean_at_most"].status == "pass"
        assert outcomes["triplet_fraction_max_at_most"].status == "pass"


class TestDeclaredGeometryBounds:
    """Count floors and radius bounds are arithmetic on the generator config.

    These are neither physics nor measurement: they follow from the radial grid
    the eval config declares. Deriving them from that config rather than
    restating a literal means changing the grid BREAKS these tests, which a
    measured tolerance would not.
    """

    def test_cusp_counts_follow_the_declared_radial_grid(
        self, thresholds: dict, radial_generator: dict
    ) -> None:
        directions = int(radial_generator["n_directions"])
        assert thresholds["cusp_finite_fit_count_min"] == directions
        assert thresholds["cusp_finite_measurement_count_min"] == (
            len(radial_generator["cusp_radii"]) * directions
        )

    def test_tail_counts_follow_the_declared_radial_grid(
        self, thresholds: dict, radial_generator: dict
    ) -> None:
        directions = int(radial_generator["n_directions"])
        assert thresholds["tail_outer_measurement_count_min"] == directions
        assert thresholds["tail_finite_measurement_count_min"] == (
            len(radial_generator["tail_radii"]) * directions
        )

    def test_radius_bounds_are_the_declared_tail_grid_endpoints(
        self, thresholds: dict, radial_generator: dict
    ) -> None:
        tail_radii = [float(radius) for radius in radial_generator["tail_radii"]]
        assert thresholds["tail_outer_radius_min_min"] == min(tail_radii)
        assert thresholds["tail_outer_radius_max_max"] == max(tail_radii)

    def test_purity_sample_floor_does_not_exceed_the_declared_draw(self) -> None:
        thresholds, _ = collect.split_gate_spec(plan.load_grid_config(GRID)["gate_spec"])
        declared = int(
            _yaml(EVAL)["evaluation_tasks"]["spatial_exchange_symmetry"]["generator"][
                "base_generator"
            ]["max_samples"]
        )
        floor = int(thresholds["triplet_fraction_finite_sample_count_min"])
        # A floor above the declared draw could never be met; one far below it
        # would accept a silently truncated diagnostic.
        assert floor <= declared
        assert floor >= declared * 0.95

    def test_diagnostic_draws_are_raised_off_the_smoke_value(self) -> None:
        tasks = _yaml(EVAL)["evaluation_tasks"]
        for name in ("spatial_exchange_symmetry", "full_model_antisymmetry"):
            draws = int(tasks[name]["generator"]["base_generator"]["max_samples"])
            # At 8 samples a reported fraction moves in steps of 0.125 and
            # cannot resolve the purity bound applied to it.
            assert draws >= 1024, name


class TestArmIsTheMeasuredOne:
    """The predeclared arm must be the shape H-C1 actually measured."""

    def test_budget_and_partition_are_declared_jointly(self, grid: dict) -> None:
        # "The a100 budget" is not well formed: the same arm supports 300k on
        # kozinsky_gpu and 100k on seas_gpu. The pair must travel together.
        assert max(grid["checkpoint_steps"]) == 300000
        assert grid["train_resources"]["partition"] == "kozinsky_gpu"
        assert grid["train_resources"]["stratum"] == "a100"

    def test_training_wall_time_clears_the_measured_projection(self, grid: dict) -> None:
        # H-C1 measured 1.7107 steps/s at c32 w4096 on a100, i.e. 2922.8
        # minutes for 300k. H-C2's checkpoint job ran 8% over its projection,
        # and under --no-requeue an overrun is a lost row.
        projected_min = 300000 / 1.7107 / 60.0
        assert grid["train_resources"]["timeout_min"] >= projected_min * 1.5
        # ... and still inside kozinsky_gpu's 7-day ceiling at margin 0.2.
        assert grid["train_resources"]["timeout_min"] <= 8064

    def test_three_seeds_and_four_chains(self, grid: dict) -> None:
        assert len(grid["seeds"]) == 3
        assert grid["eval_chains"] == 4

    def test_expanded_manifest_has_no_resume_and_a_pinned_constraint(self, grid: dict) -> None:
        rows = plan.expand_rows(grid)
        assert len(rows) == 3 + 3 * len(grid["checkpoint_steps"]) * grid["eval_chains"]
        for row in rows:
            assert row["resources"]["constraint"] == "a100"
            plan.reject_resume_overrides(row)

    def test_every_evaluated_checkpoint_is_actually_written(self, grid: dict) -> None:
        """The written cadence and the evaluated set are decoupled but not free.

        Writing costs disk and evaluating costs GPU, so the trainer writes far
        more often than the grid evaluates. The failure that decoupling creates:
        if an evaluated step is not a multiple of the write cadence it is NEVER
        WRITTEN, and its eval rows point at a checkpoint directory that does not
        exist -- 12 rows that fail inside their allocations, after the training
        row they depend on has already been paid for in full.

        THIS TEST IS NOT REDUNDANT WITH THE TERMINAL CHECKPOINT PATH. The
        Checkpoint callback writes ``step_<max_steps>`` through a terminal path
        distinct from the periodic one, so the FINAL evaluated step is written
        whatever the cadence. What this protects is every evaluated step that is
        NOT the last one -- here 100000 and 200000. Do not delete it on the
        grounds that the terminal path exists.
        """

        callbacks = _yaml(TRAIN)["callbacks"]
        checkpoint = next(
            entry for entry in callbacks if entry["_target_"] == "tpen.callback.Checkpoint"
        )
        cadence = int(checkpoint["schedule"]["every_n"])
        intermediate = [s for s in grid["checkpoint_steps"] if s != max(grid["checkpoint_steps"])]
        assert intermediate, "no non-terminal evaluated checkpoint; this test would be vacuous"
        for step in grid["checkpoint_steps"]:
            assert int(step) % cadence == 0, (
                f"evaluated checkpoint {step} is not a multiple of the write cadence "
                f"{cadence}, so it is never written and its eval rows restore nothing"
            )

    def test_keep_all_is_explicitly_configured(self) -> None:
        # Keep the compatibility key present even though TPEN ignores it.
        callbacks = _yaml(TRAIN)["callbacks"]
        checkpoint = next(
            entry for entry in callbacks if entry["_target_"] == "tpen.callback.Checkpoint"
        )
        assert "keep_last" in checkpoint
        assert checkpoint["keep_last"] is None

    def test_written_checkpoints_are_a_superset_kept_for_salvage(self, grid: dict) -> None:
        # The point of the finer cadence: a run that dies mid-budget leaves a
        # usable partial. Assert it actually writes more than it evaluates,
        # otherwise the decoupling bought nothing and the comment lies.
        callbacks = _yaml(TRAIN)["callbacks"]
        checkpoint = next(
            entry for entry in callbacks if entry["_target_"] == "tpen.callback.Checkpoint"
        )
        cadence = int(checkpoint["schedule"]["every_n"])
        written = max(grid["checkpoint_steps"]) // cadence
        assert written > len(grid["checkpoint_steps"])

    def test_train_config_max_steps_agrees_with_the_grid(self, grid: dict) -> None:
        # plan.py overrides max_steps from the grid, so a standalone run of
        # train.yaml must not answer a different question than a planned row.
        assert _yaml(TRAIN)["trainer"]["max_steps"] == max(grid["checkpoint_steps"])
