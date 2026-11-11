"""Static contracts for the minimal all-electron helium study."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.evaluation.bundle import EvaluationBundle


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "atomistic" / "he-v1"
TRAIN = STUDY / "configs" / "train.yaml"
EVAL = STUDY / "configs" / "eval.yaml"
REGISTRY = ROOT / "experiments" / "baselines" / "systems.yaml"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


#: Reserved gate-spec block holding namespace bindings, stripped before the spec
#: reaches the gates. Spelled here rather than imported for the reason below.
_METRIC_NAMESPACE_SPEC_KEY = "metric_namespaces"


def _grid_thresholds() -> dict:
    """Return the production grid's gate thresholds, bindings removed.

    Deliberately reads the YAML instead of calling the study's
    ``collect.split_gate_spec``/``plan.load_grid_config``. THE STUDY MODULES ARE
    NOT SAFELY IMPORTABLE FROM HERE, and loading them by path is not sufficient:
    ``experiments/`` holds four ``collect.py`` and three ``plan.py``, and
    ``collect.py`` reaches its siblings through BARE imports (``import plan as
    plan_stage``), so whichever study populated ``sys.modules['plan']`` first
    wins for every study afterwards. A path-based loader fixes the name this
    file imports and cannot fix the names the loaded module imports.

    Measured, not assumed: at fff76631 this test passed alone and failed in the
    composed suite with ``AttributeError: module 'plan' has no attribute
    'load_grid_config'`` -- an error naming a function that demonstrably exists,
    which reads as a broken function rather than a wrong module.

    The study-local suite still exercises the real ``split_gate_spec`` against
    this same file, so the parsing contract is covered where it can be imported
    safely; the repo-wide bare-sibling-import defect is filed separately.
    """

    payload = _load(STUDY / "configs" / "production_grid.yaml")
    spec = dict(payload["gate_spec"])
    spec.pop(_METRIC_NAMESPACE_SPEC_KEY, None)
    return spec


def _he_reference() -> float:
    systems = _load(REGISTRY)["systems"]
    helium = next(system for system in systems if system["id"] == "he_atom")
    return float(helium["reference_energy_hartree"])


def _assert_he_system(config: dict, *, include_reference_energy: bool = True) -> None:
    system = config["system"]
    assert system["id"] == "he_atom"
    assert system["n_particles"] == 2
    assert system["spin"] == {"n_up": 1, "n_down": 1}
    assert system["nuclei"] == {"positions": [[0.0, 0.0, 0.0]], "charges": [2.0]}
    if include_reference_energy:
        assert float(system["reference_energy"]) == _he_reference()
    else:
        assert "reference_energy" not in system


def _assert_factorized_model(config: dict) -> None:
    model = config["model"]
    assert "envelope" not in model
    assert "nuclear_envelope" not in model
    factors = model["factors"]
    assert [entry["_target_"] for entry in factors] == [
        "tpen.nn.ElectronElectronCusp",
        "tpen.nn.ElectronNucleusCusp",
    ]
    assert factors[1]["atoms"] == "${atoms}"
    assert config["atoms"]["_target_"] == "tpen.data.atomic_configuration.AtomicConfiguration"
    assert "GaussianConfinement" not in str(model)


def test_he_train_config_owns_batch_nuclear_context_and_coulomb_terms() -> None:
    config = _load(TRAIN)
    _assert_he_system(config, include_reference_energy=False)
    _assert_factorized_model(config)
    sampler = config["sampler"]
    assert sampler["nuclear_positions"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.positions}",
    }
    assert sampler["nuclear_charges"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.charges}",
    }
    hamiltonian_terms = config["hamiltonian_terms"]
    assert set(hamiltonian_terms) == {"kinetic", "electron_nucleus", "electron_electron", "nucleus_nucleus"}
    assert hamiltonian_terms["electron_nucleus"]["_target_"] == "tpen.physics.potential.ElectronNucleusPotential"
    assert hamiltonian_terms["electron_nucleus"]["atoms"] == "${atoms}"
    assert hamiltonian_terms["electron_nucleus"]["eps"] == 0.0
    assert hamiltonian_terms["nucleus_nucleus"] == {
        "_target_": "tpen.physics.potential.NucleusNucleusPotential",
        "atoms": "${atoms}",
    }


def test_he_eval_config_restores_same_model_and_uses_mcmc_reference_energy() -> None:
    train = _load(TRAIN)
    evaluation = _load(EVAL)
    _assert_he_system(evaluation)
    _assert_factorized_model(evaluation)
    assert evaluation["model"] == train["model"]
    assert evaluation["hamiltonian_terms"]["electron_nucleus"]["eps"] == 0.0
    assert evaluation["hamiltonian_terms"]["electron_nucleus"]["atoms"] == "${atoms}"
    assert evaluation["load"]["strict"] is True
    task = evaluation["evaluation_tasks"]["mcmc_energy"]
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    assert task["summaries"][0]["_target_"] == "tpen.evaluation.summaries.LocalEnergySummary"
    targets = [summary["_target_"] for summary in task["summaries"]]
    assert "tpen.evaluation.summaries.ReferenceEnergySummary" in targets
    reference = next(
        summary
        for summary in task["summaries"]
        if summary["_target_"] == "tpen.evaluation.summaries.ReferenceEnergySummary"
    )
    assert reference == {
        "_target_": "tpen.evaluation.summaries.ReferenceEnergySummary",
        "reference_energy": "${system.reference_energy}",
    }


def test_he_eval_replay_semantics_are_typed_and_fail_closed() -> None:
    """The driver must supply source/checkpoint identity before strict restore."""

    replay = _load(EVAL)["load"]["replay_semantics"]
    assert replay["record_schema_version"] == 1
    assert replay["source_git_sha"] == "???"
    assert replay["source_tpen_version"] == "???"
    assert replay["checkpoint_schema_version"] == "???"
    assert replay["checkpoint_kind"] == "???"
    assert replay["checkpoint_model_sha256"] == "???"
    assert replay["evaluation_config_sha256"] == "${trajectory_identity.config_sha256}"
    assert replay["runtime_dtype"] == "${runtime.dtype}"
    assert replay["reference_energy_qualification"] == (
        "infinite_nuclear_mass_nonrelativistic"
    )
    assert replay["cusp_distance"] == {
        "electron_electron_distance_form": "sqrt_squared_distance_plus_eps_squared",
        "electron_electron_distance_eps": 1.0e-12,
        "electron_electron_range_offset_form": "softplus_plus_eps",
        "electron_electron_range_offset_eps": 1.0e-12,
        "electron_nucleus_coulomb_distance_form": "euclidean_norm_clamp_min_eps",
        "electron_nucleus_coulomb_distance_eps": "${hamiltonian_terms.electron_nucleus.eps}",
    }


def test_he_eval_registers_atom_owned_radial_profiles_after_typed_contracts() -> None:
    config = _load(EVAL)
    tasks = config["evaluation_tasks"]
    radial = tasks["he_radial_profiles"]
    generator = radial["generator"]

    assert generator["_target_"] == "tpen.evaluation.generators.HeliumRadialGridGenerator"
    assert all(float(radius) > 0.0 for radius in generator["cusp_radii"])
    assert max(generator["cusp_radii"]) < min(generator["tail_radii"])
    assert generator["nuclear_positions"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.positions}",
    }
    assert generator["nuclear_charges"] == {
        "_target_": "torch.tensor",
        "data": "${system.nuclei.charges}",
    }
    assert radial["calculators"] == [
        {
            "_target_": "tpen.evaluation.calculators.ElectronNucleusRadialCalculator",
            "chunk_size": 32,
        }
    ]
    assert [summary["_target_"] for summary in radial["summaries"]] == [
        "tpen.evaluation.summaries.ElectronNucleusCuspSummary",
        "tpen.evaluation.summaries.ElectronNucleusTailSummary",
        "tpen.evaluation.summaries.ElectronNucleusRadialProfileWriter",
    ]
    assert "Hooke" not in str(radial)


def test_he_eval_separates_spatial_exchange_from_label_antisymmetry() -> None:
    config = _load(EVAL)
    tasks = config["evaluation_tasks"]
    spatial = tasks["spatial_exchange_symmetry"]

    assert spatial["generator"]["_target_"] == "tpen.evaluation.generators.ExchangeOrbitGenerator"
    assert spatial["generator"]["exchange"] == "opposite_spin_pair"
    assert spatial["calculators"] == [
        {"_target_": "tpen.evaluation.calculators.SpatialExchangeSymmetryCalculator"}
    ]
    assert tasks["full_model_antisymmetry"]["calculators"] == [
        {"_target_": "tpen.evaluation.calculators.FullModelAntisymmetryCalculator"}
    ]


def test_he_eval_wires_correlation_aware_chain_statistics() -> None:
    """A6-C is wired, inverting the pre-H-F1 "explicitly unwired" contract.

    This assertion used to require that no external chain statistics appeared,
    because no fixed-model trajectory producer existed. The producer merged and
    then sat unreferenced for a day: "A6-C done" did NOT mean "He runs have
    MCSE", and a He run at dev tip emitted none at all. H-F1 wires it, so the
    contract flips from "must be absent" to "must be present and joinable".
    """

    config = _load(EVAL)
    task = config["evaluation_tasks"]["mcmc_energy"]
    summaries = {summary["_target_"]: summary for summary in task["summaries"]}

    trajectory = summaries["tpen.evaluation.summaries.TrajectoryStatisticsSummary"]
    # The join identity admits no blanks: every field is supplied, and each
    # resolves to a `???` the driver must override rather than to a default.
    identity = config["trajectory_identity"]
    for field in ("stage", "run_id", "attempt_id", "evaluator_id", "checkpoint_file", "config_sha256"):
        assert identity[field] == "???", field
        assert trajectory[field.replace("checkpoint_file", "checkpoint_path")] == (
            "${trajectory_identity." + field + "}"
        )

    # The checkpoint FILE and the restore DIRECTORY are different paths on
    # purpose. Passing the directory where the file is wanted raises
    # IsADirectoryError inside the summary, after the chain has been sampled.
    assert trajectory["checkpoint_path"] == "${trajectory_identity.checkpoint_file}"
    assert config["load"]["path"] == "???"

    # The generator must be the trajectory one: only it publishes the explicit
    # [draw, walker] ObservableTrajectory the producer consumes.
    assert task["generator"]["_target_"] == "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    assert int(task["generator"]["n_draws"]) >= 128


#: Bundle fields supplied by the generator rather than by any calculator.
_GENERATOR_SUPPLIED_FIELDS = frozenset({"generated"})

#: Calculators whose ``name`` is not the bundle field they populate. Transform
#: calculators all write ``bundle.transform``. Asserted complete below, so a new
#: calculator cannot silently bypass the coverage check by not being listed.
#: Calculators whose bundle field is not their ``name``. Every entry here is a
#: place the coverage test below would otherwise credit a calculator with
#: producing a field it does not write.
_CALCULATOR_BUNDLE_FIELD = {
    "FullModelAntisymmetryCalculator": "transform",
    "SpatialExchangeSymmetryCalculator": "transform",
    "RotationConsistencyCalculator": "transform",
    # name="trace_equivariance", but it writes `trace_comparison`
    # (tpen/evaluation/calculators/trace.py:128).
    "TraceEquivarianceCalculator": "trace_comparison",
}


def _import_target(dotted: str) -> type:
    module_name, _, attribute = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _calculator_bundle_fields(entry: dict) -> set[str]:
    """Resolve bundle fields through any config-declared calculator delegate."""

    delegate = entry.get("calculator")
    if isinstance(delegate, dict):
        return _calculator_bundle_fields(delegate)
    calculator = _import_target(entry["_target_"])
    return {
        _CALCULATOR_BUNDLE_FIELD.get(
            calculator.__name__, getattr(calculator, "name", None)
        )
    }


def _missing_summary_fields(task: dict) -> set[str]:
    """Return summary requirements not produced by a task's calculators."""

    produced = set(_GENERATOR_SUPPLIED_FIELDS)
    for entry in task["calculators"]:
        produced.update(_calculator_bundle_fields(entry))
    required: set[str] = set()
    for entry in task["summaries"]:
        summary = _import_target(entry["_target_"])
        required.update(getattr(summary, "required_fields", frozenset()))
    return required - produced


@pytest.mark.parametrize("task_name", sorted(_load(EVAL)["evaluation_tasks"]))
def test_every_summary_field_is_produced_by_a_calculator(task_name: str) -> None:
    """No summary may consume a bundle field nothing in its task produces.

    THIS IS THE DEFECT CLASS, not one instance of it. A summary runs LAST, so a
    task missing its producer does every expensive thing correctly and then dies
    at the summary -- it cannot fail early by construction. It has now happened
    twice in this lane: `calculators: []` cost about three GPU-hours, and adding
    `SampledRecordWriter` without `WavefunctionCalculator` reproduced the same
    shape here, emitting every metric including MCSE and acceptance rate before
    failing at the record writer.

    Both instances were found by a run rather than by reading, because the
    config looks entirely reasonable either way. This test derives the property
    instead: whatever a task's summaries declare in ``required_fields`` must be
    covered by its generator plus its calculators.
    """

    task = _load(EVAL)["evaluation_tasks"][task_name]
    missing = sorted(_missing_summary_fields(task))
    assert not missing, (
        f"{task_name}: summaries require {missing}, which no calculator in "
        "this task produces. The summary runs last, so "
        "this fails only after the whole chain has been paid for."
    )


def test_summary_field_oracle_rejects_an_unproduced_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary pointed at an unknown bundle field must remain a failure."""

    summary_target = _load(EVAL)["evaluation_tasks"]["factor_response_re_equilibrated"]["summaries"][0]["_target_"]
    summary = _import_target(summary_target)
    monkeypatch.setattr(summary, "required_fields", frozenset({"not_a_bundle_field"}))

    task = _load(EVAL)["evaluation_tasks"]["factor_response_re_equilibrated"]

    assert _missing_summary_fields(task) == {"not_a_bundle_field"}


#: Positional overrides the H-F1 job scripts depend on, as (config, list key,
#: index, dotted target). These indices are addressed by NUMBER from outside the
#: repository -- ``callbacks.7.every_n_steps=1`` and
#: ``summaries.4.max_samples=64`` -- so the repo must own the fact, not a script
#: in a swept directory.
_POSITIONAL_OVERRIDE_TARGETS = (
    (TRAIN, "callbacks", 7, "tpen.callback.FactorScalars"),
    (TRAIN, "callbacks", 9, "tpen.callback.Checkpoint"),
    (EVAL, None, 4, "tpen.evaluation.summaries.SampledRecordWriter"),
)


@pytest.mark.parametrize(
    ("config_path", "list_key", "index", "dotted"),
    _POSITIONAL_OVERRIDE_TARGETS,
    ids=["callbacks.7=FactorScalars", "callbacks.9=Checkpoint", "summaries.4=SampledRecordWriter"],
)
def test_positional_override_indices_hold_the_class_the_job_scripts_assume(
    config_path: Path, list_key: str | None, index: int, dotted: str
) -> None:
    """Pin the three list positions that are addressed by number from outside.

    THE FAILURE THIS PREVENTS IS SILENT SUCCESS, not an error. The GPU pilot
    sets ``callbacks.7.every_n_steps=1`` to trace the cusp scalars every step.
    ``callbacks[6]`` is ``SamplerHealth``. Insert one callback above position 7
    and the override retunes sampler-health reporting instead, the run completes
    cleanly, and the pilot produces NO SCALAR TRACE AND NO ERROR -- on the exact
    measurement the zero-gradient-trap check depends on. The same shape applies
    to ``summaries.4.max_samples`` and to the checkpoint schedule at 9.

    Asserted against the imported CLASS rather than the dotted string, so
    renaming or relocating the class fails here too instead of leaving a string
    that matches nothing.
    """

    config = _load(config_path)
    if list_key is None:
        entries = config["evaluation_tasks"]["mcmc_energy"]["summaries"]
        where = "evaluation_tasks.mcmc_energy.summaries"
    else:
        entries = config[list_key]
        where = list_key

    assert index < len(entries), (
        f"{where} has only {len(entries)} entries, so index {index} is addressable "
        "by no override at all"
    )
    assert _import_target(entries[index]["_target_"]) is _import_target(dotted), (
        f"{where}[{index}] is {entries[index]['_target_']}, not {dotted}. A job script "
        f"overriding {where}.{index}.* now retunes the wrong component silently."
    )


def test_the_energy_task_draws_enough_per_chain_for_the_estimator_to_resolve() -> None:
    """Pin the RELATIONSHIP between configured draws and the estimator's floor.

    `produce_trajectory_statistics` refuses per chain below
    ``DEFAULT_MIN_DRAWS_PER_CHAIN``, and that refusal is correct behaviour rather
    than a fault -- it returns ``unresolved`` with a reason instead of a
    fabricated bar. But a config that drops under the floor produces an eval row
    with NO MCSE, and the visible symptom is an absent key rather than an error.

    Compared against the imported constant, never a literal 8, so the two move
    together. Production currently draws 256 per chain, a 32x margin; the
    rehearsal override in the job script sits at exactly the floor, which is why
    that script asserts the same relationship before it spends anything.
    """

    from tpen.statistics.producer import DEFAULT_MIN_DRAWS_PER_CHAIN  # noqa: PLC0415

    generator = _load(EVAL)["evaluation_tasks"]["mcmc_energy"]["generator"]
    assert generator["_target_"] == "tpen.evaluation.generators.TrajectoryMCMCGenerator"
    assert generator["n_draws"] >= DEFAULT_MIN_DRAWS_PER_CHAIN, (
        f"n_draws {generator['n_draws']} is below the estimator's per-chain floor "
        f"{DEFAULT_MIN_DRAWS_PER_CHAIN}; every eval row would report an unresolved "
        "receipt and carry no MCSE at all"
    )


def test_both_he_configs_declare_float64() -> None:
    """float64 is set in both configs and asserted, not merely present.

    The DONE clause names float64 alongside eps=0.0 and the reference literal,
    and the other two were pinned while this one was not. A local-energy
    evaluation in float32 loses the tail of the autocorrelation function to
    rounding well before the plateau, so the correlation-aware bar would be
    quietly wrong rather than absent.
    """

    for path in (TRAIN, EVAL):
        config = _load(path)
        assert config["runtime"]["dtype"] == "float64", path.name


def test_the_reference_energy_digit_string_is_pinned_in_a_test() -> None:
    """Pin the literal itself, closing the coordinated-drift hole.

    The symbolic cross-check asserts eval.yaml's `system.reference_energy`
    equals systems.yaml's `he_atom.reference_energy_hartree`, and a mutation
    kill proves it catches ONE-SIDED drift. It cannot catch a COORDINATED wrong
    change of both copies, which stays green because the two still agree with
    each other while agreeing on the wrong number.

    So the value is pinned here as well, to the digit string Aznabaev, Bekbaev
    and Korobov (arXiv:1810.11288, Table 3, attributed to Schwartz 2006) give
    for the nonrelativistic infinite-nuclear-mass helium ground state. This is a
    literal, not a computed value, so exact string equality is the right claim.

    THE COMPARISON'S VALIDITY BOUND, worth stating where the number lives: this
    reference EXCLUDES relativistic, QED and finite-mass corrections. TPEN's He
    Hamiltonian is kinetic + electron-nucleus + electron-electron at infinite
    nuclear mass, which omits them too, so the comparison is sound. A reader
    comparing against an EXPERIMENTAL helium energy would find a discrepancy
    that is physics rather than error.
    """

    expected = "-2.903724377034119598"

    registry_text = REGISTRY.read_text(encoding="utf-8")
    assert f"reference_energy_hartree: {expected}" in registry_text

    eval_text = EVAL.read_text(encoding="utf-8")
    assert f"reference_energy: {expected}" in eval_text

    # And the parsed values still agree with each other and with the literal.
    assert float(expected) == _he_reference()
    assert float(_load(EVAL)["system"]["reference_energy"]) == float(expected)


def test_every_declared_trainable_component_is_actually_trainable() -> None:
    """The three components the study trains must each SAY SO in both configs.

    THIS TEST EXISTS BECAUSE ITS ABSENCE WAS THE DEFECT. The config test already
    asserted ``ee["trainable_range"] is True`` and ``law["trainable"] is True``
    one line away, and asserted only ``channels`` for the readout. That asymmetry
    is exactly where the hole was: `PfaffianReadout` defaults to
    ``trainable: bool = False`` -- documented as keeping the weights "as fixed
    buffers for scaffold determinism" -- so passing only ``channels`` pinned
    ``w_c`` at uniform 1/32 for all 300,000 updates and the channel mixing could
    never learn.

    The failure was undetectable by reading the config, which names no default it
    disagrees with, and undetectable at runtime: under the default,
    ``channel_weights`` is ``register_parameter(..., None)`` and the value lives
    in a NON-PERSISTENT buffer, so it appears in neither ``named_parameters()``
    nor ``state_dict()``. Nothing logs it and nothing restores it. A DEFAULT THAT
    SILENTLY DISAGREES WITH THE CONFIG AUTHOR'S INTENT IS WHAT A CONFIG TEST IS
    FOR.

    Asserted in BOTH configs because enabling it adds the state-dict key
    ``readout.channel_weights`` and eval restores ``strict: true``, so a
    one-sided change fails restore in both directions.
    """

    for path in (TRAIN, EVAL):
        model = _load(path)["model"]
        readout = model["readout"]
        assert readout["_target_"] == "tpen.nn.readout.PfaffianReadout", path.name
        assert readout.get("trainable") is True, (
            f"{path.name}: PfaffianReadout.trainable defaults to False, so omitting it "
            "pins the channel weights at uniform 1/32 as a non-persistent buffer -- "
            "absent from named_parameters() and state_dict(), and unlearnable"
        )

        factors = model["factors"]
        assert factors[0]["trainable_range"] is True, path.name
        assert factors[1]["law"]["trainable"] is True, path.name

    # The two model blocks must stay SEMANTICALLY IDENTICAL WHEN PARSED, which
    # is what makes the strict restore survive this change. Deliberately NOT a
    # text comparison: the blocks differ in comment prose and always have, and
    # Hydra instantiates the parsed tree rather than the file text. The
    # assertion below is therefore the correct instrument, and describing it as
    # "byte-identical" -- as this comment and three config comments did until
    # 2026-08-17 -- claimed something stronger than the check establishes.
    assert _load(TRAIN)["model"] == _load(EVAL)["model"]


def test_the_callback_above_the_factor_scalars_index_is_still_sampler_health() -> None:
    """Names the neighbour that makes the off-by-one silent rather than loud.

    If ``callbacks[6]`` were something without an ``every_n_steps`` field, an
    off-by-one would raise and be cheap. It is ``SamplerHealth``, which accepts
    that key happily -- so the mistake costs a GPU pilot instead of a traceback.
    This test exists to keep that reasoning attached to the code.
    """

    callbacks = _load(TRAIN)["callbacks"]
    assert _import_target(callbacks[6]["_target_"]) is _import_target("tpen.callback.SamplerHealth")


def test_the_calculator_field_mapping_names_real_bundle_fields() -> None:
    """Every calculator resolves to a field `EvaluationBundle` actually has.

    THE PREVIOUS VERSION OF THIS TEST COULD NOT FAIL. Its comment named the
    hazard exactly -- "a calculator whose `name` is not its bundle field, and
    which is missing from the mapping, would be credited with producing a field
    it does not" -- and then asserted only ``hasattr(calculator, "name")``, which
    every calculator satisfies by construction. It was an instrument pointed at a
    failure it could not see, and it duly failed to see one:
    `TraceEquivarianceCalculator` has ``name = "trace_equivariance"`` and writes
    ``trace_comparison``, so the coverage test above was crediting it with a
    field that does not exist and demanding one nothing supplied.

    Checking the resolved name against `EvaluationBundle`'s real fields is what
    makes this discriminating: a bundle field is a typed attribute, so a
    fabricated one is detectable rather than merely unlikely.
    """

    bundle_fields = {field.name for field in dataclasses.fields(EvaluationBundle)}
    for task_name, task in _load(EVAL)["evaluation_tasks"].items():
        for entry in task["calculators"]:
            calculator = _import_target(entry["_target_"])
            resolved = _calculator_bundle_fields(entry)
            assert resolved <= bundle_fields, (
                f"{task_name}: {calculator.__name__} resolves to bundle field "
                f"{resolved!r}, which EvaluationBundle does not have. Either the "
                f"calculator's name differs from the field it writes and belongs in "
                f"_CALCULATOR_BUNDLE_FIELD, or the field is misspelled. Known fields: "
                f"{sorted(bundle_fields)}"
            )


def test_he_eval_energy_task_keeps_the_calculator_its_summaries_consume() -> None:
    """`calculators: []` fails at the SUMMARY stage, after all sampling is paid.

    That defect cannot fail early by construction -- it must first do every
    expensive thing correctly and then die last -- and it cost this lane about
    three GPU-hours. Pinned here so the empty list cannot come back.
    """

    task = _load(EVAL)["evaluation_tasks"]["mcmc_energy"]
    calculators = [entry["_target_"] for entry in task["calculators"]]
    assert "tpen.evaluation.calculators.LocalEnergyCalculator" in calculators
    assert "tpen.evaluation.calculators.WavefunctionCalculator" in calculators
    summaries = [entry["_target_"] for entry in task["summaries"]]
    assert "tpen.evaluation.summaries.LocalEnergySummary" in summaries


def test_he_eval_reports_sampler_health_and_retains_raw_records() -> None:
    """Two capabilities that existed, were exported, and were never named.

    `production-grid-v0` requires MCMC health beside every evaluation and raw
    [draw, walker] retention. Both were satisfiable by config alone and neither
    was configured, so an eval row carried no acceptance rate and a
    records-level run retained nothing.
    """

    config = _load(EVAL)
    task = config["evaluation_tasks"]["mcmc_energy"]
    summaries = {entry["_target_"]: entry for entry in task["summaries"]}

    assert "tpen.evaluation.summaries.SamplerStatsSummary" in summaries

    # artifact_level ALONE retains nothing: every raw-record writer returns
    # early unless the level is `records`, but each only runs if it is in a
    # task's summaries list. Setting the level without the writer satisfies the
    # contract's letter and keeps no records at all.
    assert config["evaluation"]["artifact_level"] == "records"
    writer = summaries["tpen.evaluation.summaries.SampledRecordWriter"]

    # The writer's declared capacity is the complete streamed draw-by-walker
    # grid. A smaller capacity is a task error, never a silent head slice.
    draws = int(task["generator"]["n_draws"])
    walkers = int(config["evaluation_sampler"]["n_walkers"])
    assert int(writer["max_samples"]) == draws * walkers
    assert writer["include_term_energies"] is True


def test_he_eval_sampler_carries_the_predeclared_stride_and_burn_in() -> None:
    sampler = _load(EVAL)["evaluation_sampler"]
    # Stride 20 is a DECLARED TIE-BREAK on MCSE and inflation, not a throughput
    # winner: ESS/second could not distinguish 10 from 20 at the measured noise
    # floor. Burn-in 100 was demonstrated sufficient ON THE MEAN.
    assert sampler["n_steps"] == 20
    assert sampler["burn_in"] == 100
    assert sampler["proposal_scale"] == 0.5


def test_he_eval_invariant_tasks_use_mcmc_batches_that_preserve_nuclear_context() -> None:
    config = _load(EVAL)
    expected_generators = {
        "full_model_antisymmetry": "tpen.evaluation.generators.PermutationOrbitGenerator",
        "spatial_exchange_symmetry": "tpen.evaluation.generators.ExchangeOrbitGenerator",
        "trace_equivariance": "tpen.evaluation.generators.PermutationOrbitGenerator",
    }
    for name, expected_generator in expected_generators.items():
        generator = config["evaluation_tasks"][name]["generator"]
        assert generator["_target_"] == expected_generator
        assert generator["base_generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"


def test_he_train_targets_instantiate_and_consume_the_same_nuclear_context() -> None:
    """Resolve Hydra targets through one actual nuclear-context forward pass."""

    config = OmegaConf.load(TRAIN)
    model = instantiate(config.model)
    # Runner owns the configured floating-point policy.  Mirror that launch
    # boundary for this direct target-instantiation contract test.
    model.to(dtype=torch.float64)
    sampler = instantiate(config.sampler)
    terms = instantiate(config.hamiltonian_terms)
    assert sampler.nuclear_positions is not None
    assert sampler.nuclear_charges is not None
    assert torch.equal(sampler.nuclear_positions, torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.equal(sampler.nuclear_charges, torch.tensor([2.0], dtype=torch.float64))
    batch = ElectronBatch(
        positions=torch.tensor([[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]], dtype=torch.float64),
        spins=torch.tensor([[1, -1]]),
        nuclear_positions=sampler.nuclear_positions,
        nuclear_charges=sampler.nuclear_charges,
    )
    output = model(batch)
    output.validate(batch_size=batch.batch_size)
    assert set(terms) == {"kinetic", "electron_nucleus", "electron_electron", "nucleus_nucleus"}
    from tpen.physics.hamiltonian import local_energy

    result = local_energy(terms, model, batch, return_terms=True)
    assert torch.equal(result.terms["nucleus_nucleus"], torch.zeros_like(result.total))

    # A5 ownership resolution: the wavefunction's ElectronNucleusCusp and the
    # Hamiltonian's ElectronNucleusPotential/NucleusNucleusPotential each
    # instantiate their own `AtomicConfiguration` from the same declarative
    # `${atoms}` config source -- Hydra's recursive `instantiate` re-runs a
    # `_target_` for every tree position it appears at, so this is NOT one
    # shared Python object. Assert both facts explicitly: the objects are
    # distinct, and no code path anywhere relies on them being identical --
    # every Hamiltonian term instead reconciles construction-time `atoms`
    # against transported batch context by explicit value comparison
    # (`_validate_batch_atoms_context`, exercised above via `local_energy`).
    en_cusp = next(f for f in model.factors if type(f).__name__ == "ElectronNucleusCusp")
    hamiltonian_en_atoms = terms["electron_nucleus"].atoms
    hamiltonian_nn_atoms = terms["nucleus_nucleus"].atoms
    assert en_cusp.atoms is not hamiltonian_en_atoms
    assert en_cusp.atoms is not hamiltonian_nn_atoms
    assert hamiltonian_en_atoms is not hamiltonian_nn_atoms
    assert en_cusp.atoms == hamiltonian_en_atoms
    assert en_cusp.atoms == hamiltonian_nn_atoms


def test_he_production_width_reproduces_the_calibrated_mapping() -> None:
    """The measured capacity numbers describe THIS model, not a similar one.

    H-C1 measured 38921.1 MB and 1.7107 steps/s for "c32" using its own
    `apply_width` mapping. A different spelling of c32 -- most easily by leaving
    `hidden_channels` behind -- would leave those numbers perfectly true about
    H-C1's model and perfectly false about the production arm, with no test
    catching it and the capacity argument still reading as fully cited.
    """

    channels = 32
    for config in (_load(TRAIN), _load(EVAL)):
        model = config["model"]
        assert model["embedding"]["out_channels"] == channels
        # The one knob most easily forgotten. `hidden_channels = channels * 4`
        # is the mapping H-C1 actually ran; the pre-production 4/16 pair was
        # already this same ratio.
        assert model["embedding"]["hidden_channels"] == channels * 4
        assert model["readout"]["channels"] == channels
        for layer in model["layers"]:
            assert layer["mixing"]["channels"] == channels
            assert layer["path_aggregation"]["channels"] == channels


def test_he_configs_enable_both_trainable_cusp_ranges_identically() -> None:
    train = _load(TRAIN)["model"]["factors"]
    evaluation = _load(EVAL)["model"]["factors"]
    # Enabling the trainable law adds exactly two state-dict keys, so a strict
    # model-only restore fails in BOTH directions across the change. A
    # train/eval mismatch is not a degraded run, it is a run that cannot start.
    assert train == evaluation

    ee = next(f for f in train if f["_target_"] == "tpen.nn.ElectronElectronCusp")
    assert ee["trainable_range"] is True

    en = next(f for f in train if f["_target_"] == "tpen.nn.ElectronNucleusCusp")
    law = en["law"]
    assert law["_target_"] == "tpen.nn.CurvatureElectronNucleusCuspLaw"
    assert law["trainable"] is True
    # NONZERO on purpose. At exactly c = 0 the gradient with respect to the
    # range parameter d is identically zero, so a defaults-instantiated
    # trainable range cannot move on step one -- and the law's own defaults
    # (trainable=True, curvature_coefficient=0.0) land in that trap.
    assert float(law["curvature_coefficient"]) != 0.0
    assert float(law["curvature_range"]) > 0.0


def test_he_tail_band_brackets_the_law_s_own_outer_tail_slope() -> None:
    """The tail gate is centered on `outer_tail_slope`, never on a hardcoded -Z.

    At c = 0 the law is exactly `-Z r` and the expected slope is -Z. The moment
    c != 0 it is `-Z + c/d`, so a band calibrated against the pure linear law
    must not be applied unchanged. This asserts the predeclared band actually
    contains the value the configured law returns, rather than a value someone
    assumed it returns.
    """

    thresholds = _grid_thresholds()

    law = instantiate(OmegaConf.create(_load(TRAIN)["model"]["factors"][1]["law"]))
    charges = torch.tensor(_load(TRAIN)["system"]["nuclei"]["charges"], dtype=torch.float64)
    expected = float(law.outer_tail_slope(charges).reshape(-1)[0].item())

    # The configured initialization must sit inside the band the gate declares,
    # or the gate would fail a correctly initialized model on step one.
    assert thresholds["tail_outer_slope_mean_min"] <= expected
    assert expected <= thresholds["tail_outer_slope_mean_max"]
    # And it must NOT be -Z: asserting that keeps this test honest about the
    # fact that the curved law shifted the center at all.
    assert expected != -float(charges[0].item())
    assert abs(expected - (-float(charges[0].item()))) < 0.1

    # A decaying tail requires c/d < Z. The law does not enforce it; the gate's
    # sign requirement is the executable proxy, so it must demand every ray.
    assert thresholds["tail_negative_slope_fraction_min"] == 1.0


def test_trainable_law_breaks_strict_restore_in_both_directions() -> None:
    """The predeclared checkpoint break, asserted rather than discovered.

    H-C2's pilot checkpoint was trained with the linear law, so it cannot be
    restored under the production config and its measured tau describes a
    different model. Pinning the break here means a future config change that
    silently reverts the law fails a test instead of failing a restore inside an
    allocation.
    """

    from tpen.nn import ElectronNucleusCusp, CurvatureElectronNucleusCuspLaw

    atoms_cfg = OmegaConf.create(_load(TRAIN)["atoms"])
    OmegaConf.update(atoms_cfg, "positions.data", _load(TRAIN)["system"]["nuclei"]["positions"])
    OmegaConf.update(atoms_cfg, "charges.data", _load(TRAIN)["system"]["nuclei"]["charges"])
    atoms = instantiate(atoms_cfg)

    linear = ElectronNucleusCusp(atoms)
    trainable = ElectronNucleusCusp(
        atoms,
        law=CurvatureElectronNucleusCuspLaw(
            curvature_coefficient=0.01, curvature_range=1.0, trainable=True
        ),
    )

    # Exactly two keys, and the default law contributes none -- which is why
    # A5's bit-identity guarantee holds until a config actually switches.
    assert linear.state_dict() == {}
    assert sorted(trainable.state_dict()) == [
        "law.raw_curvature_coefficient",
        "law.raw_curvature_range",
    ]

    with pytest.raises(RuntimeError):
        linear.load_state_dict(trainable.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        trainable.load_state_dict(linear.state_dict(), strict=True)


def test_he_train_enables_canonical_host_wall_cost_callback_at_config_root() -> None:
    """Compose the concrete resource callback in the training config."""

    train_callbacks = _load(TRAIN)["callbacks"]
    concrete_target = "tpen.callback.ResourceUsage"
    assert sum(entry["_target_"] == concrete_target for entry in train_callbacks) == 1
    resource_usage = next(entry for entry in train_callbacks if entry["_target_"] == concrete_target)
    assert resource_usage["allocator_probe"] == {
        "_target_": "tpen.accelerator.TorchAllocatorPeakProbe",
        "device": "${runtime.device}",
    }


def test_he_eval_enables_canonical_host_wall_cost_callbacks_at_config_root() -> None:
    """Use typed evaluation boundaries without claiming broken device sync."""

    config = _load(EVAL)
    concrete_target = "tpen.callback.ResourceUsage"
    assert sum(entry["_target_"] == concrete_target for entry in config["callbacks"]) == 1
    callbacks = {
        entry["_target_"]: entry
        for entry in config["callbacks"]
    }
    for target in (
        "tpen.callback.EvaluationTiming",
        "tpen.callback.EvaluationComponentTiming",
        "tpen.callback.DiagnosticTiming",
    ):
        assert callbacks[target]["accelerator_synchronize"] is False
    resource_usage = callbacks[concrete_target]
    assert resource_usage["allocator_probe"] == {
        "_target_": "tpen.accelerator.TorchAllocatorPeakProbe",
        "device": "${runtime.device}",
    }
    assert "callbacks" not in config["runner"]
