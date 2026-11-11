"""Integration test: the Train runner executes a VMC smoke loop end-to-end.

Drives the full configured path -- ``run_from_config`` -> ``Train`` runner ->
``make_optimizer`` -> ``VMCTrainer.fit`` -> sampler -> Hamiltonian terms ->
surrogate loss -> optimizer step -> loggers/callbacks -- and asserts the
standard run artifacts and finite ``train`` metrics. No convergence assertions.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tpen.checkpoint import resolve_checkpoint_dir
from tpen.checkpoint.hashing import file_sha256
from tpen.run import run_from_config

FIXTURE = Path(__file__).resolve().parents[1] / "artifacts" / "training" / "vmc_smoke.yaml"

ALLOWED_NONFINITE_KEYS = {"energy_stderr"}

# Every durable phase key the wired `TrainPhaseTiming` reports for one completed
# training iteration. Each name is owned by a concrete `TrainingPhase` type, so
# this tuple pins the public spelling of the whole `train/perf` phase surface.
PHASE_TIMING_KEYS = (
    "sampling_time_sec",
    "batch_build_time_sec",
    "local_energy_time_sec",
    "forward_time_sec",
    "objective_time_sec",
    "backward_time_sec",
    "optimizer_step_time_sec",
    "post_step_metrics_time_sec",
)


def _run(tmp_path: Path, cfg: DictConfig | None = None):
    """Drive one configured run under `tmp_path` and return its run directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Run root. Each call needs its own, because the returned directory is
        found by globbing and exactly one run dir is expected under it.
    cfg : omegaconf.DictConfig or None, optional
        Prepared config. ``None`` loads the fixture unchanged.
    """

    if cfg is None:
        cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(FIXTURE), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("vmc_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def test_train_runner_writes_standard_artifacts(tmp_path) -> None:
    run_dir = _run(tmp_path)

    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.csv",
        "metrics.jsonl",
        "run_start.json",
        "checkpoints/latest.json",
        # Cadence 2 writes step 2; train_end still writes terminal step 3.
        "checkpoints/step_000002/COMPLETE",
        # Checkpoint steps use the resume cursor, so a 3-step run ends at step 3.
        "checkpoints/step_000003/manifest.json",
        "checkpoints/step_000003/model.pt",
        "checkpoints/step_000003/COMPLETE",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["hardware"]["hostname"]
    assert "cpu_count_logical" in metadata["hardware"]
    assert "cuda_available" in metadata["hardware"]
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["runtime"]["dtype"] == "float64"
    assert "python_version" in metadata["runtime"]
    assert "slurm" in metadata

    # Three attempted iterations, each of which applied its optimizer update.
    trainer_state = json.loads((run_dir / "checkpoints/step_000003/trainer.json").read_text())
    assert set(trainer_state) == {
        "next_iteration",
        "completed_updates",
        "parameter_layout",
        "nonfinite_local_energy_policy",
    }
    assert trainer_state["next_iteration"] == 3
    assert trainer_state["completed_updates"] == 3
    # The ACTIVE estimator, recorded beside the numbers it produced. Pinned to
    # the literal rather than to the module default: an assertion that reads
    # the default and compares it to itself passes for ANY default, including
    # one nobody chose -- which is the failure the policy key exists to catch.
    assert trainer_state["nonfinite_local_energy_policy"] == "mask"

    # The v2 manifest names both counters instead of one ambiguous `step`, and
    # the directory the run wrote is the one `next_iteration` names.
    manifest = json.loads((run_dir / "checkpoints/step_000003/manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "tpen.checkpoint"
    assert "step" not in manifest
    assert manifest["next_iteration"] == 3
    assert manifest["completed_updates"] == 3
    assert "spenn_version" not in manifest["provenance"]
    assert manifest["provenance"]["tpen_version"]

    latest = json.loads((run_dir / "checkpoints/latest.json").read_text())
    assert latest["checkpoint_dir"] == "step_000003"
    assert latest["step"] == 3
    assert resolve_checkpoint_dir(run_dir / "checkpoints") == run_dir / "checkpoints/step_000003"


def test_train_runner_logs_finite_train_metrics(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_records = [record["metrics"] for record in records if record.get("namespace") == "train"]
    sampler_records = [record["metrics"] for record in records if record.get("namespace") == "train/sampler"]
    perf_records = [record["metrics"] for record in records if record.get("namespace") == "train/perf"]
    runtime_records = [record["metrics"] for record in records if record.get("namespace") == "runtime"]
    assert len(train_records) == 3, "expected one train record per step"
    assert len(sampler_records) == 3, "expected one train/sampler record per step"
    # Two callbacks write `train/perf`: TrainStepTiming reports whole-step wall
    # time at `step_end`, TrainPhaseTiming the typed phase breakdown at
    # `TrainingIterationCompleted`. Split them by key rather than by position.
    step_timing_records = [record for record in perf_records if "step_time_sec" in record]
    phase_timing_records = [record for record in perf_records if "step_time_sec" not in record]
    assert len(step_timing_records) == 3, "expected one step-timing record per step"
    assert len(phase_timing_records) == 3, "expected one phase-timing record per step"
    assert any("wall_time_sec" in record for record in runtime_records)

    last = train_records[-1]
    for key in (
        "loss",
        "energy",
        "energy_variance",
        "local_energy_n_finite",
        "local_energy_finite_fraction",
        "logabs_mean",
    ):
        assert key in last, f"missing metric: {key}"
    # The physical training estimator is logged as `energy`, never `energy_mean`.
    assert "energy_mean" not in last
    assert not any(key.startswith("sampler.") for key in last)
    assert "acceptance_rate" in sampler_records[-1]
    assert "n_walkers" in sampler_records[-1]
    assert "step_time_sec" in step_timing_records[-1]
    assert "step_time_sec_rolling_mean" in step_timing_records[-1]

    # This is the only test that drives several phase types through the real
    # RunContext -> _dispatch_occurrence -> Callback.handle_occurrence path, so
    # pin the exact key set and require finite durations, not mere presence.
    for record in phase_timing_records:
        assert set(record) == set(PHASE_TIMING_KEYS), f"unexpected phase keys: {sorted(record)}"
        for key in PHASE_TIMING_KEYS:
            value = record[key]
            assert isinstance(value, (int, float)), f"non-numeric phase metric {key}={value!r}"
            assert math.isfinite(value), f"non-finite phase metric {key}={value}"

    # JSONL serialization with allow_nan=False would already have failed the run
    # on any non-finite value; assert finiteness directly for good measure.
    for record in train_records:
        for key, value in record.items():
            if key in ALLOWED_NONFINITE_KEYS or not isinstance(value, (int, float)):
                continue
            assert math.isfinite(value), f"non-finite metric {key}={value}"


# --------------------------------------------------------------------------
# Resume equivalence: train(2N) == train(N) + resume(N)
# --------------------------------------------------------------------------
#
# The Event Clock refactor's central promise is that a resumed run continues
# correctly, and until this test nothing exercised it end to end: every
# `train_resume` test in the suite drives trainer/sampler stubs, so none of them
# can observe whether restored state reproduces real training arithmetic. A
# prior receipt compared which STEPS each metric namespace fired on across an
# uninterrupted and a restored arm -- cadence-gate correctness -- and never
# compared VALUES.
#
# Both arms carry the SAME `trainer` block, and that is a hard constraint rather
# than a stylistic one: `train_resume` verifies the manifest's `trainer_config`
# hash, which covers `max_steps`, so a first arm run at `max_steps=3` cannot
# supply the resume source for a `max_steps=6` continuation -- the restore is
# refused at the hash gate before any weight is read. The interruption is
# therefore expressed by resuming from the uninterrupted arm's OWN mid-run
# checkpoint, which asserts the same property: continuing from a persisted step
# must reproduce the run that never stopped.
EQUIVALENCE_MAX_STEPS = 6
RESUME_STEP = 3


def _equivalence_config(
    *,
    load: dict[str, str] | None = None,
    save_rng: bool | None = None,
) -> DictConfig:
    """Return the smoke config extended to a resumable ``max_steps=6`` run.

    The fixture's periodic cadence (``schedule.every_n: 2``) counts applied
    optimizer updates, so at ``max_steps=6`` it writes steps 2, 4 and 6 and
    never produces the mid-run checkpoint this test resumes from. Widening it to
    `RESUME_STEP` writes ``step_000003`` and ``step_000006`` instead. Nothing
    else about the fixture changes, so the resumed arm's restore passes every
    component hash the manifest carries.

    Parameters
    ----------
    load : dict or None, optional
        Restore config attached to the runner. ``None`` leaves the runner
        without one, which is ``mode: none``.
    save_rng : bool or None, optional
        Override for the checkpoint stream. ``None`` keeps the explicit
        ``TrainResume`` payload and its default ``True``. A supplied value
        selects the flag-owned component set so the negative arm can create a
        deliberately partial checkpoint without conflicting with that profile.
    """

    cfg = OmegaConf.load(FIXTURE)
    cfg.trainer.max_steps = EQUIVALENCE_MAX_STEPS
    for callback in cfg.callbacks:
        if callback.get("_target_") != "tpen.callback.Checkpoint":
            continue
        # The composed stream produces the resume source on its periodic path;
        # its terminal path remains unwindowed by the schedule.
        if callback.get("periodic", True):
            callback.schedule.every_n = RESUME_STEP
        if save_rng is not None:
            callback.payload = None
            callback.save_rng = save_rng
    if load is not None:
        cfg.runner.load = load
    return cfg


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:06d}"


def _train_metric_lines(run_dir: Path) -> list[tuple[int, str]]:
    """Return ``(step, raw JSONL line)`` for every ``train``-namespace record.

    The raw line is kept rather than the parsed mapping so equality is byte
    equality. `tpen.logging.JSONL` writes ``{step, namespace, event, metrics}``
    with sorted keys and no timestamp, so two runs that computed the same
    numbers emit the same bytes.
    """

    lines = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("namespace") == "train":
            lines.append((int(record["step"]), line))
    return lines


def _occurrences(run_dir: Path, name: str) -> list[dict]:
    """Return typed occurrence records whose event type ends with `name`."""

    path = run_dir / "occurrences.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [record for record in records if record["event"].endswith(name)]


def _model_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    return torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)


def _diverged_parameters(left: Path, right: Path) -> list[str]:
    """Return the parameter names whose tensors are not bitwise equal.

    `torch.equal` rather than `torch.allclose`: a resume that reconstitutes
    state correctly reproduces the same float64 arithmetic on the same inputs,
    so any tolerance would hide exactly the drift this test exists to catch.
    """

    left_state = _model_tensors(left)
    right_state = _model_tensors(right)
    assert set(left_state) == set(right_state), "checkpoints hold different parameter sets"
    return [name for name, tensor in left_state.items() if not torch.equal(tensor, right_state[name])]


@pytest.fixture(scope="module")
def uninterrupted_run(tmp_path_factory) -> Path:
    """Arm A: one uninterrupted ``max_steps=6`` run, shared by every arm below.

    Module-scoped so each resume arm restores from the identical baseline
    artifacts, which is what makes the negative arms single-variable mutations
    rather than two runs that differ in an unknown number of ways.
    """

    return _run(tmp_path_factory.mktemp("uninterrupted"), _equivalence_config())


def test_resume_reproduces_the_uninterrupted_run_bitwise(uninterrupted_run, tmp_path) -> None:
    """Arm B: restoring arm A's step-3 checkpoint reaches arm A's step-6 state.

    This is the property `TODO.md` names and that no test has ever asserted:
    ``train(2N)`` equals ``train(N)`` + ``resume(N)``, compared on values rather
    than on which steps fired.
    """

    source = _checkpoint_dir(uninterrupted_run, RESUME_STEP)
    assert (source / "COMPLETE").exists(), "arm A wrote no resume source"

    resumed_run = _run(
        tmp_path,
        _equivalence_config(load={"mode": "train_resume", "path": str(source)}),
    )

    final_a = _checkpoint_dir(uninterrupted_run, EQUIVALENCE_MAX_STEPS)
    final_b = _checkpoint_dir(resumed_run, EQUIVALENCE_MAX_STEPS)
    assert _diverged_parameters(final_a, final_b) == []

    # Both durable counters, from the terminal checkpoint of each arm.
    expected_progress = {
        "next_iteration": EQUIVALENCE_MAX_STEPS,
        "completed_updates": EQUIVALENCE_MAX_STEPS,
    }
    trainer_a = json.loads((final_a / "trainer.json").read_text())
    trainer_b = json.loads((final_b / "trainer.json").read_text())
    expected_keys = {*expected_progress, "parameter_layout", "nonfinite_local_energy_policy"}
    assert set(trainer_a) == expected_keys
    assert set(trainer_b) == expected_keys
    assert trainer_a == trainer_b
    assert {key: trainer_a[key] for key in expected_progress} == expected_progress
    # Both arms must have run under the SAME estimator, and under the one
    # intended. `trainer_a == trainer_b` already forces agreement between the
    # arms; this pins WHICH policy they agreed on, so a run that silently
    # switched estimators cannot satisfy a bitwise-resume test by having both
    # halves switch together.
    assert trainer_a["nonfinite_local_energy_policy"] == "mask"

    # The resumed arm logs only the steps it actually ran, and every one of them
    # is byte-identical to the same step in the run that never stopped.
    resumed_lines = _train_metric_lines(resumed_run)
    assert [step for step, _ in resumed_lines] == list(range(RESUME_STEP, EQUIVALENCE_MAX_STEPS))
    uninterrupted_tail = [
        line for step, line in _train_metric_lines(uninterrupted_run) if step >= RESUME_STEP
    ]
    assert [line for _, line in resumed_lines] == uninterrupted_tail


def test_training_resume_records_restored_checkpoint_identity(uninterrupted_run, tmp_path) -> None:
    """A training resume must reach ``occurrences.jsonl``, not only the strings.

    `tpen.runner.Evaluate` emits the typed `CheckpointRestored` at its restore
    and `tpen.runner.Train` did not, so a training resume recorded its restored
    checkpoint identity in ``events.jsonl`` alone. ``completed_updates`` is the
    field that matters: it names which model version the run continues.
    """

    source = _checkpoint_dir(uninterrupted_run, RESUME_STEP)
    resumed_run = _run(
        tmp_path,
        _equivalence_config(load={"mode": "train_resume", "path": str(source)}),
    )

    restored = _occurrences(resumed_run, "CheckpointRestored")
    assert len(restored) == 1
    report = restored[0]["fields"]["report"]
    assert report["mode"] == "train_resume"
    assert report["checkpoint_dir"] == str(source)
    assert report["next_iteration"] == RESUME_STEP
    assert report["completed_updates"] == RESUME_STEP
    assert report["loaded_model"] is True
    assert report["loaded_optimizer"] is True
    assert report["loaded_trainer"] is True
    assert report["loaded_sampler"] is True
    assert report["loaded_rng"] is True

    # A run that restored nothing records nothing, so the record above is the
    # restore's own, not an artifact every run happens to write.
    assert _occurrences(uninterrupted_run, "CheckpointRestored") == []


def test_resume_is_refused_when_the_checkpoint_carries_no_rng_state(tmp_path) -> None:
    """Negative arm 1: ``save_rng=False`` makes ``train_resume`` fail loudly.

    Asserted as a direct `pytest.raises` on the real exception type rather than
    as an ``xfail``: the expected outcome is one specific refusal, which is an
    ordinary assertion about behaviour. An ``xfail`` would accept ANY failure,
    including an unrelated one, and would report an eventual regression as an
    unexpected pass rather than as a failed assertion.

    What this arm proves and what it does not: a checkpoint written without
    ``rng.pt`` has no ``files['rng']`` manifest entry, so the restore is refused
    at `tpen.checkpoint.restore._required_file` BEFORE any component is loaded.
    It therefore pins the fail-loud contract, but it cannot show that the
    resumed arithmetic depends on restored randomness -- no arithmetic runs. The
    arm that shows that is below.
    """

    source_run = _run(tmp_path / "rngless", _equivalence_config(save_rng=False))
    source = _checkpoint_dir(source_run, RESUME_STEP)
    assert not (source / "rng.pt").exists()

    cfg = _equivalence_config(load={"mode": "train_resume", "path": str(source)})
    cfg.run.root = str(tmp_path / "refused")
    with pytest.raises(FileNotFoundError, match="lacks file entry 'rng'"):
        run_from_config(cfg, config_path=str(FIXTURE), command="pytest", raise_exceptions=True)


def test_resume_diverges_when_the_restored_sampler_stream_is_perturbed(
    uninterrupted_run, tmp_path
) -> None:
    """Negative arm 2: the bitwise match above is load-bearing, not vacuous.

    Every Markov-chain draw is taken with ``generator=self._generator`` (ADR-013:
    randomness is component-owned), so the sampler's persisted generator state --
    not the process RNG globals in ``rng.pt`` -- is the stream the continued
    iterations actually consume. Reseeding exactly that one field in a COPY of
    arm A's checkpoint, leaving every other byte alone, must move the final
    weights. If it does not, the model consumes no restored randomness and the
    equality asserted above would prove nothing.

    Asserted as a positive statement of inequality rather than as an inverted or
    ``xfail``-marked run of the positive test, for the same reason as arm 1: the
    expected outcome here is a passing assertion about divergence, and a test
    that is expected to pass should be spelled as one.
    """

    perturbed = tmp_path / "perturbed" / f"step_{RESUME_STEP:06d}"
    shutil.copytree(_checkpoint_dir(uninterrupted_run, RESUME_STEP), perturbed)
    sampler_state = torch.load(perturbed / "sampler.pt", map_location="cpu", weights_only=False)
    generator = torch.Generator(device=sampler_state["generator_device"])
    generator.manual_seed(20260811)
    sampler_state["generator_state"] = generator.get_state()
    torch.save(sampler_state, perturbed / "sampler.pt")
    manifest_path = perturbed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"]["sampler_sha256"] = file_sha256(perturbed / "sampler.pt")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed_run = _run(
        tmp_path / "diverged",
        _equivalence_config(load={"mode": "train_resume", "path": str(perturbed)}),
    )

    diverged = _diverged_parameters(
        _checkpoint_dir(uninterrupted_run, EQUIVALENCE_MAX_STEPS),
        _checkpoint_dir(resumed_run, EQUIVALENCE_MAX_STEPS),
    )
    assert diverged, "a reseeded sampler stream changed nothing; the equivalence test is vacuous"
