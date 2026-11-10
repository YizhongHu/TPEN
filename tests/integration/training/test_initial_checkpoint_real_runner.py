"""Real-runner behavioral probes for `tpen.callback.InitialCheckpoint`.

PR 610, review round 1, tests-only worker. These probes complement, and do not
duplicate, the committed suite (`tests/unit/callback/test_initial_checkpoint.py`,
`tests/unit/training/test_initial_checkpoint_integration.py`): every test here
drives the real configured path -- ``run_from_config`` -> ``tpen.runner.Train``
-> ``VMCTrainer.fit`` -- rather than calling the callback or the trainer
directly, and compares against a SEPARATELY run or SEPARATELY constructed
reference rather than against the callback's own write.

The acceptance contract (TO item 5cfad8d2, note ``acceptance-contract``)
requires: "A resumed ... lineage is refused." The callback's own refusal check
(`tpen/callback/initial_checkpoint.py`) reads only the first occurrence's
``event.operation.step`` and the trainer's ``next_iteration`` /
``completed_updates`` counters. A lineage resumed from a checkpoint that was
itself written at the zero boundary carries all three values at zero, so
nothing in that check distinguishes "fresh" from "resumed-but-still-at-zero".
``test_initial_checkpoint_refuses_a_resumed_zero_counter_lineage`` below is
expected to surface exactly that gap; it is written as a direct assertion, not
an ``xfail``, so a current failure is preserved as real evidence rather than
hidden. The zero-counter checkpoint it resumes from is written by a SEPARATE
``InitialCheckpoint``-enabled run under the IDENTICAL ``trainer`` config the
resumed run uses: ``train_resume`` verifies the manifest's ``trainer_config``
hash (`tpen/checkpoint/restore.py`), which covers ``trainer.max_steps``, so a
source written under a *different* ``max_steps`` (e.g. a genuine
``max_steps=0`` run) would be refused at that hash gate before ever reaching
`InitialCheckpoint`'s own check -- a harness defect, not the finding this test
exists to pin.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tpen.data import Walkers
from tpen.run import run_from_config
from tpen.sampling.metropolis import MetropolisSampler

FIXTURE = Path(__file__).resolve().parents[1] / "artifacts" / "training" / "vmc_smoke.yaml"

_INITIAL_CHECKPOINT_CALLBACK = {
    "_target_": "tpen.callback.InitialCheckpoint",
    "output_dir": "${run.dir}/checkpoints",
}


def _cfg(
    *,
    max_steps: int,
    with_initial_checkpoint: bool = False,
    initial_checkpoint_first: bool = False,
    load: dict[str, str] | None = None,
) -> DictConfig:
    """The committed `vmc_smoke` fixture, narrowed to one trainer step.

    Parameters
    ----------
    max_steps : int
        Overrides ``trainer.max_steps``.
    with_initial_checkpoint : bool, optional
        Append (or prepend) an `InitialCheckpoint` callback sharing the
        periodic `Checkpoint` callback's ``output_dir``, per the acceptance
        contract's "same keep-all directory" requirement.
    initial_checkpoint_first : bool, optional
        Place `InitialCheckpoint` before the periodic `Checkpoint` callback
        instead of after it, to check the coexistence path is independent of
        configured callback order.
    load : dict or None, optional
        ``runner.load`` restore spec, e.g. ``{"mode": "train_resume", "path": ...}``.
    """

    cfg = OmegaConf.load(FIXTURE)
    cfg.trainer.max_steps = max_steps
    if with_initial_checkpoint:
        entry = OmegaConf.create(_INITIAL_CHECKPOINT_CALLBACK)
        if initial_checkpoint_first:
            cfg.callbacks.insert(0, entry)
        else:
            cfg.callbacks.append(entry)
    if load is not None:
        cfg.runner.load = load
    return cfg


def _run(tmp_path: Path, cfg: DictConfig) -> Path:
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(FIXTURE), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("vmc_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def _checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:06d}"


def _occurrences(run_dir: Path, name: str) -> list[dict]:
    """Return typed occurrence records whose event type ends with `name`."""

    path = run_dir / "occurrences.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [record for record in records if record["event"].endswith(name)]


def _train_metric_lines(run_dir: Path) -> list[str]:
    return [
        line
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip() and '"namespace": "train"' in line
    ]


# --------------------------------------------------------------------------
# Tensor/structural state comparison, independent-of-shape.
# --------------------------------------------------------------------------


def _assert_tensor_dict_equal(left: dict, right: dict, *, label: str) -> None:
    assert set(left) == set(right), f"{label}: key sets differ ({set(left)} vs {set(right)})"
    for key, left_value in left.items():
        right_value = right[key]
        assert torch.equal(left_value, right_value), f"{label}.{key} diverged"


def _assert_model_equal(left_dir: Path, right_dir: Path) -> None:
    left = torch.load(left_dir / "model.pt", map_location="cpu", weights_only=False)
    right = torch.load(right_dir / "model.pt", map_location="cpu", weights_only=False)
    _assert_tensor_dict_equal(left, right, label="model")


def _assert_optional_tensor_equal(left, right, *, label: str) -> None:
    if left is None or right is None:
        assert left is None and right is None, f"{label}: one side is None, the other is not"
        return
    assert torch.equal(left, right), f"{label} diverged"


def _assert_aux_equal(left: dict, right: dict) -> None:
    """Compare `Walkers.aux` key by key: values may be tensors or plain Python.

    A bare ``left == right`` on the whole dict raises "the truth value of a
    tensor is ambiguous" the moment any value is a tensor (observed on Cannon:
    `aux` carries at least one tensor alongside plain floats), the same class
    of defect the sampler-state comparator above exists to avoid.
    """

    assert set(left) == set(right), f"walkers.aux: key sets differ ({set(left)} vs {set(right)})"
    for key, left_value in left.items():
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), f"walkers.aux.{key} diverged"
        else:
            assert left_value == right_value, f"walkers.aux.{key} diverged"


def _assert_walkers_equal(left: Walkers | None, right: Walkers | None) -> None:
    if left is None or right is None:
        assert left is None and right is None, "walkers: one side is None, the other is not"
        return
    assert isinstance(left, Walkers) and isinstance(right, Walkers)
    _assert_optional_tensor_equal(left.positions, right.positions, label="walkers.positions")
    _assert_optional_tensor_equal(left.logabs, right.logabs, label="walkers.logabs")
    _assert_optional_tensor_equal(left.sign, right.sign, label="walkers.sign")
    _assert_optional_tensor_equal(left.spins, right.spins, label="walkers.spins")
    assert left.atomic_configuration == right.atomic_configuration, "walkers.atomic_configuration diverged"
    _assert_aux_equal(left.aux, right.aux)


def _assert_sampler_equal(left_dir: Path, right_dir: Path) -> None:
    """Compare `MetropolisSampler.mcmc_state_dict()` payloads field by field.

    Not a generic dict-of-tensors comparison: ``walkers`` holds a `Walkers`
    dataclass whose tensor fields make a bare ``==`` raise "the truth value of
    a tensor is ambiguous", and ``atomic_configuration``/``has_burned_in``/
    ``acceptance_rate``/``generator_device`` are plain Python values, not
    tensors.
    """

    left = torch.load(left_dir / "sampler.pt", map_location="cpu", weights_only=False)
    right = torch.load(right_dir / "sampler.pt", map_location="cpu", weights_only=False)
    assert set(left) == set(right), f"sampler state keys differ ({set(left)} vs {set(right)})"
    _assert_walkers_equal(left["walkers"], right["walkers"])
    assert left["atomic_configuration"] == right["atomic_configuration"]
    assert left["has_burned_in"] == right["has_burned_in"]
    assert left["acceptance_rate"] == right["acceptance_rate"]
    assert torch.equal(left["generator_state"], right["generator_state"]), "sampler.generator_state diverged"
    assert left["generator_device"] == right["generator_device"]


# --------------------------------------------------------------------------
# RNG snapshot, taken immediately before the first `collect_samples` call.
# --------------------------------------------------------------------------


def _run_with_pre_sample_rng_snapshot(
    tmp_path: Path, cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    """Run `cfg` and return (run_dir, RNG state snapshotted before first sampling).

    Wraps `MetropolisSampler.collect_samples` to RECORD -- never mutate -- the
    torch, Python `random`, NumPy, and sampler-generator state on its first
    invocation, then calls the original method unchanged. `monkeypatch.undo()`
    restores the class before returning, so a second call in the same test
    wraps the real method again rather than stacking wrappers.
    """

    snapshot: dict[str, object] = {}
    original = MetropolisSampler.collect_samples

    def _recording_collect_samples(self, model, *, reset: bool = False, device=None):
        if not snapshot:
            snapshot["torch_rng_state"] = torch.get_rng_state().clone()
            snapshot["python_random_state"] = random.getstate()
            snapshot["numpy_random_state"] = np.random.get_state()
            snapshot["sampler_generator_state"] = self._generator.get_state().clone()
        return original(self, model, reset=reset, device=device)

    monkeypatch.setattr(MetropolisSampler, "collect_samples", _recording_collect_samples)
    try:
        run_dir = _run(tmp_path, cfg)
    finally:
        monkeypatch.undo()
    assert snapshot, "collect_samples was never called -- no pre-sample RNG state was captured"
    return run_dir, snapshot


def _assert_rng_snapshots_equal(left: dict[str, object], right: dict[str, object]) -> None:
    assert set(left) == set(right)
    assert torch.equal(left["torch_rng_state"], right["torch_rng_state"]), "torch RNG state diverged"
    assert left["python_random_state"] == right["python_random_state"], "Python random state diverged"
    left_numpy, right_numpy = left["numpy_random_state"], right["numpy_random_state"]
    assert left_numpy[0] == right_numpy[0]
    assert np.array_equal(left_numpy[1], right_numpy[1]), "NumPy RNG state array diverged"
    assert left_numpy[2:] == right_numpy[2:]
    assert torch.equal(
        left["sampler_generator_state"], right["sampler_generator_state"]
    ), "sampler generator state diverged before the first sample"


# --------------------------------------------------------------------------
# Probe 1: a resumed lineage must be refused even when every counter is zero.
# --------------------------------------------------------------------------


def test_initial_checkpoint_accepts_a_fresh_independent_lineage(tmp_path) -> None:
    """Positive control: a genuinely fresh lineage is accepted, as elsewhere.

    Exercised here (not only in the unit/integration suites already committed)
    so the refusal probe below has a same-fixture, same-callback-wiring control
    proving the callback fires and writes under this exact configuration.
    """

    run_dir = _run(tmp_path, _cfg(max_steps=1, with_initial_checkpoint=True))
    assert (_checkpoint_dir(run_dir, 0) / "COMPLETE").is_file()
    assert (_checkpoint_dir(run_dir, 1) / "COMPLETE").is_file()


def test_initial_checkpoint_refuses_a_resumed_zero_counter_lineage(tmp_path) -> None:
    """A lineage resumed from a zero-boundary checkpoint must still be refused.

    Arm A (the source) is an ``InitialCheckpoint``-enabled ``max_steps=1`` run:
    its ``step_000000`` checkpoint is a genuine zero-counter checkpoint written
    under the SAME ``trainer`` config Arm B will use, so Arm B's restore is not
    refused by the unrelated ``trainer_config`` hash check.

    Arm B loads that checkpoint with ``mode: train_resume`` into fresh objects,
    under a DIFFERENT, empty checkpoint directory, with `InitialCheckpoint`
    enabled and ``max_steps=1``. The restore succeeding is asserted FIRST and
    explicitly (``CheckpointRestored``'s ``loaded_*`` fields), so the premise
    -- a resumed lineage, not some unrelated failure -- is verified independent
    of the refusal check below. Arm B's restored trainer counters and the
    first `Started(TrainingIteration)` occurrence's step are then all zero --
    exactly the state a genuinely fresh lineage has. The acceptance contract
    requires refusal of a resumed lineage regardless; the callback's own check
    only inspects those zero-valued counters, so this is expected to currently
    NOT raise, which is the finding this test exists to pin.
    """

    source_run = _run(tmp_path / "zero-source", _cfg(max_steps=1, with_initial_checkpoint=True))
    zero_source = _checkpoint_dir(source_run, 0)
    assert (zero_source / "COMPLETE").is_file()

    resumed_cfg = _cfg(
        max_steps=1,
        with_initial_checkpoint=True,
        load={"mode": "train_resume", "path": str(zero_source)},
    )
    resumed_cfg.run.root = str(tmp_path / "resumed")
    raised: ValueError | None = None
    try:
        run_from_config(
            resumed_cfg, config_path=str(FIXTURE), command="pytest", raise_exceptions=True
        )
    except ValueError as exc:
        raised = exc

    resumed_dirs = list((tmp_path / "resumed").glob("vmc_smoke/*/*"))
    assert len(resumed_dirs) == 1, f"expected one run dir, found {resumed_dirs}"
    resumed_dir = resumed_dirs[0]

    restored = _occurrences(resumed_dir, "CheckpointRestored")
    assert len(restored) == 1, "the restore itself must have been attempted and recorded"
    report = restored[0]["fields"]["report"]
    assert report["mode"] == "train_resume"
    assert report["next_iteration"] == 0
    assert report["completed_updates"] == 0
    assert report["loaded_model"] is True
    assert report["loaded_optimizer"] is True
    assert report["loaded_trainer"] is True
    assert report["loaded_sampler"] is True
    assert report["loaded_rng"] is True

    assert raised is not None, (
        "InitialCheckpoint must refuse a RESUMED lineage even when every counter "
        "reads zero (next_iteration=completed_updates=0 above, restore confirmed "
        "successful); the current check only inspects those counters and the "
        "first iteration's step, so a zero-counter resume is indistinguishable "
        "from a fresh lineage and is wrongly accepted"
    )
    assert "fresh zero-update lineage" in str(raised)
    assert not (_checkpoint_dir(resumed_dir, 0) / "COMPLETE").is_file()


# --------------------------------------------------------------------------
# Probe 2: the acceptance comparator missing from the committed suite.
# --------------------------------------------------------------------------


def test_initial_checkpoint_matches_a_separately_run_zero_step_checkpoint(tmp_path) -> None:
    """`InitialCheckpoint`'s write matches an independent real-runner reference.

    The committed integration test compares the written checkpoint to a model
    built by calling the test helper directly (`build_tiny_spenn()`), never to
    another pass through the actual runner. This test is the comparator the
    acceptance contract names: a SEPARATE ``max_steps=0`` run through
    ``run_from_config`` (periodic `Checkpoint` only, no `InitialCheckpoint`,
    and no restore involved, so no config-hash coupling to the other arm)
    must write the bitwise-identical model and sampler state that an
    `InitialCheckpoint`-enabled ``max_steps=1`` run writes at its own
    update-zero boundary. Both runs are independent Hydra instantiations from
    the same fixture and the same fixed seed; neither reads the other's
    output as an expected value.
    """

    reference_run = _run(tmp_path / "reference", _cfg(max_steps=0))
    reference_zero = _checkpoint_dir(reference_run, 0)

    initial_run = _run(tmp_path / "initial", _cfg(max_steps=1, with_initial_checkpoint=True))
    initial_zero = _checkpoint_dir(initial_run, 0)

    _assert_model_equal(reference_zero, initial_zero)
    _assert_sampler_equal(reference_zero, initial_zero)


# --------------------------------------------------------------------------
# Probe 3: RNG / first-update noninterference.
# --------------------------------------------------------------------------


def test_initial_checkpoint_does_not_perturb_the_first_update(tmp_path, monkeypatch) -> None:
    """Presence of `InitialCheckpoint` must not change subsequent training.

    Two otherwise-identical one-step runs, one with `InitialCheckpoint` wired
    in and one without, must (a) reach the first `collect_samples` call with
    byte-identical torch/Python-random/NumPy/sampler-generator state, and (b)
    land on bitwise-identical model and sampler state at the terminal
    (update-1) checkpoint, with byte-identical ``train`` metric records. Train
    metric non-emptiness is asserted explicitly so neither comparison can pass
    vacuously on two runs that produced nothing.
    """

    without_dir, without_snapshot = _run_with_pre_sample_rng_snapshot(
        tmp_path / "without", _cfg(max_steps=1), monkeypatch
    )
    with_dir, with_snapshot = _run_with_pre_sample_rng_snapshot(
        tmp_path / "with", _cfg(max_steps=1, with_initial_checkpoint=True), monkeypatch
    )

    _assert_rng_snapshots_equal(without_snapshot, with_snapshot)

    _assert_model_equal(_checkpoint_dir(without_dir, 1), _checkpoint_dir(with_dir, 1))
    _assert_sampler_equal(_checkpoint_dir(without_dir, 1), _checkpoint_dir(with_dir, 1))

    without_metrics = _train_metric_lines(without_dir)
    with_metrics = _train_metric_lines(with_dir)
    assert without_metrics, "control arm produced no train metrics -- comparison would be vacuous"
    assert without_metrics == with_metrics


# --------------------------------------------------------------------------
# Probe 4: coexistence with the periodic Checkpoint callback across a real update.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "initial_checkpoint_first", [False, True], ids=["ic-after-periodic", "ic-before-periodic"]
)
def test_initial_checkpoint_and_periodic_checkpoint_coexist_across_an_update(
    tmp_path, initial_checkpoint_first
) -> None:
    """Both the zero and the first-update directories exist, in either callback order.

    A ``max_steps=1`` run with both callbacks writes `step_000000` (from
    `InitialCheckpoint`, before any sample or update) AND `step_000001` (from
    the periodic `Checkpoint` callback's terminal write, after the one real
    update). `step_000000`'s content must be untouched by the later update --
    checked here against the probe-2 reference run's independently-written
    zero checkpoint -- rather than merely checked for existence.
    """

    reference_run = _run(
        tmp_path / f"reference-{initial_checkpoint_first}", _cfg(max_steps=0)
    )
    reference_zero = _checkpoint_dir(reference_run, 0)

    run_dir = _run(
        tmp_path / f"coexist-{initial_checkpoint_first}",
        _cfg(
            max_steps=1,
            with_initial_checkpoint=True,
            initial_checkpoint_first=initial_checkpoint_first,
        ),
    )

    zero_dir = _checkpoint_dir(run_dir, 0)
    one_dir = _checkpoint_dir(run_dir, 1)
    assert (zero_dir / "COMPLETE").is_file()
    assert (one_dir / "COMPLETE").is_file()
    _assert_model_equal(reference_zero, zero_dir)
    _assert_sampler_equal(reference_zero, zero_dir)


# --------------------------------------------------------------------------
# Probe 5: a requeue-style resume into the SAME keep-all directory completes,
# with BOTH InitialCheckpoint and ParameterActivity a verified no-op against
# the SAME reused paths -- never a late write, never a mislabeled record.
# --------------------------------------------------------------------------


_PARAMETER_ACTIVITY_NAME = "parameter_activity.json"


_REQUEUE_MAX_STEPS = 2


def _requeue_cfg(*, run_id: str, load: dict[str, str] | None = None) -> DictConfig:
    """A fixed-`run_id`, fixed-`max_steps` config, so two invocations share one
    keep-all directory AND one `trainer_config` hash -- a genuine requeue
    reuses the identical trainer config; only its `runner.load` differs."""

    cfg = OmegaConf.load(FIXTURE)
    cfg.trainer.max_steps = _REQUEUE_MAX_STEPS
    cfg.run.run_id = run_id
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            # Default cadence (`every_n: 2`) only writes the terminal step at
            # `max_steps=2`; `every_n: 1` also writes step 1 as a genuine
            # intermediate checkpoint to resume arm B from.
            callback.schedule.every_n = 1
    cfg.callbacks.append(OmegaConf.create(_INITIAL_CHECKPOINT_CALLBACK))
    cfg.callbacks.append(
        OmegaConf.create(
            {
                "_target_": "tpen.callback.ParameterActivity",
                "output_path": "${run.dir}/" + _PARAMETER_ACTIVITY_NAME,
            }
        )
    )
    if load is not None:
        cfg.runner.load = load
    return cfg


def test_requeue_into_the_same_directory_completes_with_a_verified_no_op(tmp_path) -> None:
    """A step > 0 resume into the SAME output directory completes end to end.

    Arm A is a fresh, ``max_steps=2`` run (periodic cadence widened to
    ``every_n: 1`` so ``step_000001`` is a genuine intermediate checkpoint,
    not only the terminal one), fixed to a known ``run.run_id`` so its
    ``run.dir`` (and therefore both callbacks' shared output paths) is
    reused, not a fresh per-attempt directory, by arm B below -- the requeue
    shape this follow-up targets, not the independent-directory restore the
    probes above pin.

    Arm B resumes from arm A's INTERMEDIATE (``step_000001``) checkpoint,
    reusing arm A's exact ``run.run_id`` (so its ``run.dir``) AND its exact
    ``max_steps`` (so the restore's ``trainer_config`` hash matches -- a
    genuine requeue reuses the identical trainer config, it does not widen
    it). So arm B trains exactly one more update (to ``max_steps=2``), and
    its restored trainer counters (asserted below via ``CheckpointRestored``)
    are nonzero, with its first iteration/applied-update both reading a
    nonzero step. `InitialCheckpoint` and `ParameterActivity` must each
    become a VERIFIED no-op against their SAME, reused output paths --
    neither refuses, and neither writes anything new; `ParameterActivity` in
    particular must never record arm B's own later-update gradients as
    though they were the lineage's first.
    """

    run_id = "requeue-same-directory"
    source_dir = _run(tmp_path, _requeue_cfg(run_id=run_id))
    zero_dir = _checkpoint_dir(source_dir, 0)
    one_dir = _checkpoint_dir(source_dir, 1)
    assert (zero_dir / "COMPLETE").is_file()
    assert (one_dir / "COMPLETE").is_file()
    zero_model_before = (zero_dir / "model.pt").read_bytes()
    inventory_path = source_dir / _PARAMETER_ACTIVITY_NAME
    assert inventory_path.is_file()
    inventory_before = inventory_path.read_bytes()
    inventory_mtime_before = inventory_path.stat().st_mtime_ns

    resumed_cfg = _requeue_cfg(run_id=run_id, load={"mode": "train_resume", "path": str(one_dir)})
    resumed_dir = _run(tmp_path, resumed_cfg)
    assert resumed_dir == source_dir

    restored = _occurrences(resumed_dir, "CheckpointRestored")
    assert len(restored) == 1
    report = restored[0]["fields"]["report"]
    assert report["next_iteration"] == 1
    assert report["completed_updates"] == 1

    # InitialCheckpoint's verified no-op: the shared `step_000000` checkpoint
    # is untouched by the resumed arm.
    assert (zero_dir / "model.pt").read_bytes() == zero_model_before
    # ParameterActivity's verified no-op: the shared inventory is untouched,
    # never overwritten with arm B's own (later) first-applied-update record.
    assert inventory_path.read_bytes() == inventory_before
    assert inventory_path.stat().st_mtime_ns == inventory_mtime_before
