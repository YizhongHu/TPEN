"""DS-A0 Accelerate bounded spike: gates A-G0, A-G1, A-G1b, A-G2.

Facility-agnostic by construction: selection is on ``probe_gloo_capability()``
and on whether ``accelerate`` imports. No facility name appears anywhere in this
module. A missing capability produces an explicit skip naming the capability, and
per the acceptance contract A SKIP SATISFIES NO GATE -- which is precisely why
A-G0 exists and why it ASSERTS rather than skips.

Scored against ``tests.helpers.vmc_scientific_oracle``, the same oracle the
native spike used, over the same fixtures, so a difference in outcome between the
two spikes is attributable to the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.helpers.ddp_capability import probe_gloo_capability
from tests.helpers.ddp_subprocess_harness import HarnessBounds, run_gloo_subprocess_group
from tests.helpers.vmc_scientific_oracle import (
    loss_tolerance_envelope,
    naive_per_shard_clip_then_sum,
    oracle_global_clip,
    oracle_vmc_objective,
    summation_error_envelope,
)
from tests.spikes.accelerate_ddp.fixtures import scientific_fixture
from tests.spikes.accelerate_ddp.model_access import SemanticWavefunction
from tests.spikes.accelerate_ddp.statistics import local_centered_objective

WORKER_MODULE = "tests.spikes.accelerate_ddp.worker"
_BOUNDS = HarnessBounds(process_group_timeout=60.0, watchdog_timeout=120.0)


def _accelerate_capability() -> tuple[bool, str]:
    """Report whether Accelerate is importable, and why not when it is not."""

    try:
        import accelerate  # noqa: F401
    except ModuleNotFoundError as exc:
        return False, f"accelerate is not installed: {exc}"
    return True, ""


def _require_capabilities() -> None:
    """Skip with a NAMED capability, never a bare skip."""

    gloo = probe_gloo_capability()
    if not gloo.sufficient:
        pytest.skip(f"missing capability: gloo/subprocess -- {dict(gloo.reasons)}")
    available, reason = _accelerate_capability()
    if not available:
        pytest.skip(f"missing capability: accelerate -- {reason}")


def _run_scenario(world_size: int, scenario: str, tmp_path: Path) -> tuple:
    """Launch one scenario and return the harness result plus per-rank states."""

    result = run_gloo_subprocess_group(
        world_size=world_size,
        fault_plan=None,
        bounds=_BOUNDS,
        tmp_path=tmp_path,
        worker_module=WORKER_MODULE,
        worker_extra_args=["--scenario", scenario],
    )
    invocation = Path(result.invocation_dir)
    states = []
    for rank in range(world_size):
        state_path = invocation / f"state_{rank}.json"
        # A missing state artifact is a hard failure, not a skip: it means the
        # rank died before it could record anything, and no gate may be credited
        # from an absent observation.
        assert state_path.exists(), (
            f"rank {rank} wrote no state artifact; exit_codes={result.exit_codes}, "
            f"logs under {invocation}"
        )
        states.append(json.loads(state_path.read_text()))
    return result, states


def _oracle_for(world_size: int, kind: str):
    """Recompute the whole iteration in ONE process over the concatenation.

    This is the reference the distributed run must equal. It builds the same
    module, evaluates every shard through it, and lets the oracle do the global
    reduction -- shards are passed separately rather than pre-concatenated so the
    oracle's merge algebra is exercised rather than bypassed.
    """

    model = SemanticWavefunction()
    feature_shards = []
    energy_shards = []
    for rank in range(world_size):
        features, energy = scientific_fixture(world_size, rank, kind=kind)
        feature_shards.append(features)
        energy_shards.append(energy)
    logabs_shards = [model(features) for features in feature_shards]
    oracle = oracle_vmc_objective(logabs_shards, energy_shards)
    return model, feature_shards, energy_shards, logabs_shards, oracle


# --- A-G0 -------------------------------------------------------------------


def test_a_g0_accelerate_capability_is_present_and_recorded() -> None:
    """A-G0: the capability was PRESENT, and its identity is on the record.

    ASSERTS rather than skips, deliberately. A-G0 exists because ``accelerate``
    is an optional extra that nothing installs by default, so a suite lacking it
    would skip every arm and report green. A gate that could only ever skip could
    never go red, which is the vacuous shape it exists to rule out.

    Its red arm is an ENVIRONMENT mutation, not a source mutation: it is
    demonstrated by running in an environment built without the extra, so it
    cannot ride the revert-byte-identically discipline the source mutants use.
    Observed red on Cannon job 44954901.
    """

    gloo = probe_gloo_capability()
    assert gloo.sufficient, f"gloo/subprocess capability insufficient: {dict(gloo.reasons)}"

    import accelerate

    assert accelerate.__version__, "accelerate must report a version"
    # Import the exact surfaces the other gates stand on, so a rename surfaces
    # here rather than midway through a scientific comparison.
    from accelerate import Accelerator, InitProcessGroupKwargs  # noqa: F401
    from accelerate.utils import broadcast_object_list, gather_object  # noqa: F401

    assert Path(accelerate.__file__).exists()


# --- A-G1 -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("world_size", "kind"),
    [(2, "regular"), (3, "m2")],
    ids=["world2-regular", "world3-m2-uneven"],
)
def test_a_g1_matches_concatenated_oracle(world_size: int, kind: str, tmp_path: Path) -> None:
    """A-G1: the distributed step equals the single-process concatenation.

    Both world sizes the contract names, including the M2 uneven fixture whose
    raw counts are [5, 3, 7] and finite counts [3, 0, 5] -- a topology with an
    entirely nonfinite shard, which is where a reducer that averages rank-local
    means diverges from one that merges sufficient statistics.
    """

    _require_capabilities()
    result, states = _run_scenario(world_size, "score-step", tmp_path)

    assert result.exit_codes == tuple([0] * world_size), (
        f"ranks did not all succeed: {result.exit_codes}"
    )
    assert result.all_reaped and not result.watchdog_fired
    for rank, state in enumerate(states):
        assert state["ok"], f"rank {rank} failed: {state.get('failure_message')}"

    model, feature_shards, energy_shards, logabs_shards, oracle = _oracle_for(world_size, kind)

    # Global metrics: every rank must publish the SAME global values, and they
    # must equal the oracle's. Rank agreement is checked first because a
    # disagreement there is a different defect from a shared wrong answer.
    for rank, state in enumerate(states):
        stats = state["result"]["stats"]
        assert stats["finite_count"] == oracle.metrics["local_energy_n_finite"], (
            f"rank {rank} disagrees on the global finite count"
        )
        assert stats["total_count"] == oracle.metrics["local_energy_n_total"], (
            f"rank {rank} disagrees on the global raw count"
        )
        energy_terms = torch.cat([shard[torch.isfinite(shard)] for shard in energy_shards])
        assert stats["mean"] == pytest.approx(
            oracle.metrics["energy"], abs=summation_error_envelope(energy_terms)
        ), f"rank {rank} global mean differs from the oracle"

    # The PUBLISHED loss is the global scientific loss, never the world-size
    # compensated surrogate that was actually backpropagated. Assert both: that
    # the published value matches the oracle, and that it is NOT the surrogate.
    all_terms = torch.cat(
        [
            (torch.where(torch.isfinite(e), e - oracle.metrics["energy"], torch.zeros_like(e)) * l).detach()
            for e, l in zip(energy_shards, logabs_shards, strict=True)
        ]
    )
    loss_tolerance = loss_tolerance_envelope(all_terms)
    for rank, state in enumerate(states):
        published = state["result"]["global_loss"]
        assert published == pytest.approx(oracle.metrics["loss"], abs=loss_tolerance), (
            f"rank {rank} published loss differs from the oracle"
        )
        surrogate = state["result"]["local_surrogate_loss"]
        assert surrogate != pytest.approx(published, abs=loss_tolerance) or world_size == 1, (
            f"rank {rank} published the W-scaled surrogate instead of the global loss"
        )

    # Parameter gradients: the reference is autograd through the SAME module over
    # the concatenation, not a hand-derived formula, so this checks the whole
    # chain rather than the score-function identity alone.
    oracle.loss.backward()
    reference_gradients = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
    }
    gradient_tolerance = 3.0 * summation_error_envelope(all_terms)
    for rank, state in enumerate(states):
        observed = state["result"]["gradients"]
        assert set(observed) == set(reference_gradients), (
            f"rank {rank} reported gradients for {set(observed)}, expected "
            f"{set(reference_gradients)}"
        )
        for name, reference in reference_gradients.items():
            got = torch.tensor(observed[name], dtype=torch.float64)
            expected = reference.flatten()
            assert torch.allclose(got, expected, atol=gradient_tolerance, rtol=0.0), (
                f"rank {rank} gradient {name}: {got.tolist()} != {expected.tolist()}"
            )

    # Post-step parameters: every rank must hold identical parameters after the
    # update, which is what makes the replicas still one model.
    first = states[0]["result"]["post_step_parameters"]
    for rank, state in enumerate(states[1:], start=1):
        assert state["result"]["post_step_parameters"] == first, (
            f"rank {rank} diverged from rank 0 after the optimizer step"
        )

    # THE ACCELERATE-SPECIFIC HAZARD, closed explicitly rather than assumed:
    # `Accelerator.backward` divides by `gradient_accumulation_steps`. A hidden
    # 1/n there would stack on the reducer's average and corrupt every comparison
    # above while every arm still passed.
    for rank, state in enumerate(states):
        assert state["result"]["gradient_accumulation_steps"] == 1, (
            f"rank {rank} ran with gradient accumulation, which scales the loss"
        )

    # A-E2's boundary, observed here because the run already produced it: exactly
    # one prepared forward per update, and coordinate work on the raw module.
    for rank, state in enumerate(states):
        counts = state["result"]["forward_counts"]
        assert counts["prepared"] == 1, (
            f"rank {rank} ran {counts['prepared']} prepared forwards, expected exactly 1"
        )
        assert state["result"]["model_provenance"]["unwrap_returns_original_identity"] is True


# --- A-G1b ------------------------------------------------------------------


def test_a_g1b_local_centering_disagrees_with_the_global_oracle() -> None:
    """A-G1b: rank-local centering must FAIL, demonstrated not asserted.

    Process-free on purpose. The claim is about the ARITHMETIC -- that centering
    on rank-local means is a different function from centering on the global mean
    -- and running it through subprocesses would add cost without adding
    evidence. The minimal counterexample is one distinct sample per rank, which
    is the smallest input on which the two disagree at all.
    """

    logabs_shards = [
        torch.tensor([1.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([2.0], dtype=torch.float64, requires_grad=True),
    ]
    energy_shards = [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
    ]

    oracle = oracle_vmc_objective(logabs_shards, energy_shards)
    wrong = local_centered_objective(logabs_shards, energy_shards)

    # Each rank centering on its own single sample gives exactly zero, while the
    # global centering does not. Non-vacuity is the point: this must be a real
    # disagreement, not two numbers that happen to be close.
    assert float(wrong.item()) == 0.0
    assert abs(float(oracle.loss.item())) > 1e-6, (
        "the counterexample must produce a nonzero global objective, or the "
        "comparison would pass for the wrong reason"
    )


def test_a_g1b_per_shard_clipping_disagrees_with_global_clipping() -> None:
    """A-G1b: per-shard-clip-then-sum must FAIL against a global clip.

    Large, nearly opposing per-shard gradients whose global sum is small. Clipping
    each shard first destroys the cancellation, so the two answers differ by far
    more than any rounding envelope.
    """

    shard_gradients = [
        torch.tensor([100.0, 0.0], dtype=torch.float64),
        torch.tensor([-99.0, 0.0], dtype=torch.float64),
    ]
    clip_norm = 1.0

    correct = oracle_global_clip(shard_gradients, clip_norm)
    naive = naive_per_shard_clip_then_sum(shard_gradients, clip_norm)

    assert float(correct.norm().item()) <= clip_norm + 1e-12
    assert not torch.allclose(correct, naive, atol=1e-6), (
        "per-shard clipping must disagree with global clipping on this fixture"
    )


# --- A-G2 -------------------------------------------------------------------


def test_a_g2_global_zero_finite_count_refuses_before_backward(tmp_path: Path) -> None:
    """A-G2 (M4): global M == 0 refuses on every rank, with no gradient event.

    The load-bearing assertion is the parameter-gradient EVENT COUNT, not
    unchanged-looking parameters. Unchanged state is compatible with a backward
    that ran and cancelled; a zero event count is not. This follows DS-N's own
    R2-F2 repair rather than repeating the weaker check it replaced.
    """

    _require_capabilities()
    world_size = 2
    result, states = _run_scenario(world_size, "all-invalid", tmp_path)

    assert result.exit_codes == tuple([0] * world_size), (
        f"the refusal path must exit cleanly on every rank: {result.exit_codes}"
    )
    assert result.all_reaped and not result.watchdog_fired

    for rank, state in enumerate(states):
        assert state["ok"], f"rank {rank} failed: {state.get('failure_message')}"
        observed = state["result"]
        assert observed["stats"]["finite_count"] == 0, (
            f"rank {rank} did not see a global finite count of zero"
        )
        # ONE COMMON refusal: every rank, not just the one whose shard was empty.
        assert observed["refused_before_backward"] is True, (
            f"rank {rank} did not refuse before backward"
        )
        assert observed["parameter_gradient_events"] == 0, (
            f"rank {rank} recorded {observed['parameter_gradient_events']} parameter-gradient "
            "events; the refusal must precede any backward"
        )
        assert observed["gradient_reductions"] == 0, (
            f"rank {rank} performed {observed['gradient_reductions']} gradient reductions"
        )
        assert observed["prepared_forwards"] == 0, (
            f"rank {rank} ran a prepared forward before refusing"
        )
        assert observed["parameters_unchanged"] is True
        assert observed["optimizer_state_empty"] is True, (
            f"rank {rank} advanced optimizer state despite refusing"
        )


# --- A-G3 / A-G3b -----------------------------------------------------------


def _run_phase(world_size: int, tmp_path: Path, checkpoint_root: Path, *extra: str) -> tuple:
    """Run one resume phase as its OWN harness invocation.

    Separate invocations are the point of the gate: every process is genuinely
    torn down between phases. An in-process reload would be a much weaker claim
    than the contract makes, and would pass even if nothing were persisted.
    """

    result = run_gloo_subprocess_group(
        world_size=world_size,
        fault_plan=None,
        bounds=_BOUNDS,
        tmp_path=tmp_path,
        worker_module=WORKER_MODULE,
        worker_extra_args=["--scenario", "resume", "--checkpoint-root", str(checkpoint_root), *extra],
    )
    invocation = Path(result.invocation_dir)
    states = []
    for rank in range(world_size):
        state_path = invocation / f"state_{rank}.json"
        assert state_path.exists(), f"rank {rank} wrote no state; exit_codes={result.exit_codes}"
        states.append(json.loads(state_path.read_text()))
    return result, states


def test_a_g3_teardown_and_resume_equals_continuous(tmp_path: Path) -> None:
    """A-G3: K+L continuous == K, checkpoint, tear down ALL processes, restore, L.

    EXACTLY, not approximately: the same arithmetic in the same order on the same
    inputs must reproduce bit-identical parameters and optimizer momentum. A
    tolerance here would hide precisely the kind of partial restore this gate
    exists to catch.
    """

    _require_capabilities()
    world_size = 2
    k, ell = 2, 3

    continuous_root = tmp_path / "ckpt-continuous"
    split_root = tmp_path / "ckpt-split"

    _, continuous = _run_phase(
        world_size, tmp_path, continuous_root,
        "--phase", "continuous", "--k", str(k), "--l", str(ell),
    )
    _, first = _run_phase(
        world_size, tmp_path, split_root,
        "--phase", "first-half", "--k", str(k), "--l", str(ell),
    )
    _, second = _run_phase(
        world_size, tmp_path, split_root,
        "--phase", "second-half", "--k", str(k), "--l", str(ell),
    )

    for label, states in (("continuous", continuous), ("first", first), ("second", second)):
        for rank, state in enumerate(states):
            assert state["ok"], f"{label} rank {rank} failed: {state.get('failure_message')}"

    for rank in range(world_size):
        assert second[rank]["result"]["completed_updates_restored"] == k, (
            f"rank {rank} restored the wrong update counter"
        )
        # Canonical model state carries NO `module.` prefix: the keys are the raw
        # semantic module's, not the wrapper's.
        keys = second[rank]["result"]["canonical_model_keys"]
        assert keys and not any(key.startswith("module.") for key in keys), (
            f"rank {rank} canonical keys carry a wrapper prefix: {keys}"
        )
        # Rank-local sampler state must come back RANK-DISTINCT, not broadcast.
        walkers = second[rank]["result"]["sampler_walkers"]
        assert walkers == [rank * 100.0 + i for i in range(4)], (
            f"rank {rank} restored walkers {walkers}, which are not its own"
        )
        assert second[rank]["result"]["sampler_proposal_count"] == 7 + rank

        assert second[rank]["result"]["parameters"] == continuous[rank]["result"]["parameters"], (
            f"rank {rank}: resumed parameters differ from the continuous run"
        )
        assert (
            second[rank]["result"]["optimizer_momentum"]
            == continuous[rank]["result"]["optimizer_momentum"]
        ), f"rank {rank}: resumed optimizer momentum differs from the continuous run"


def test_a_g3b_perturbing_one_rank_sidecar_makes_resume_fail(tmp_path: Path) -> None:
    """A-G3b: the resume must FAIL when one rank's sampler shard is perturbed.

    Non-vacuity for A-G3. The perturbation is verified to have actually changed
    the bytes before the reload is attempted -- a no-op edit would make this arm
    pass while demonstrating nothing.
    """

    _require_capabilities()
    world_size = 2
    root = tmp_path / "ckpt-perturbed"

    _run_phase(world_size, tmp_path, root, "--phase", "first-half", "--k", "2", "--l", "3")

    sidecar = root / "generations" / "gen-000001" / "sidecars" / "rank-00001.json"
    assert sidecar.exists(), "the published generation must contain rank 1's sidecar"
    before = sidecar.read_bytes()
    payload = json.loads(before)
    payload["sampler_state"]["proposal_count"] = 999999
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    assert sidecar.read_bytes() != before, "the perturbation must actually change the bytes"

    _, states = _run_phase(world_size, tmp_path, root, "--phase", "second-half", "--k", "2", "--l", "3")

    # Rank 1 must refuse. Fail-closed on a digest mismatch is a STRONGER outcome
    # than a silently different result, and it is what the store implements.
    assert states[1]["ok"] is False, "rank 1 accepted a perturbed sidecar"
    assert states[1]["failure_type"] == "CheckpointCorrupt", (
        f"rank 1 failed for the wrong reason: {states[1].get('failure_type')} "
        f"{states[1].get('failure_message')}"
    )


# --- A-G5 -------------------------------------------------------------------


def test_a_g5_failed_shard_writer_leaves_no_selectable_generation(tmp_path: Path) -> None:
    """A-G5: a rank dying mid-save must leave NO selectable COMPLETE generation.

    Checked from the PARENT against the filesystem, not from a worker's report: a
    worker that died cannot testify, and the claim is about what a later reader
    would find on disk.
    """

    _require_capabilities()
    world_size = 2
    root = tmp_path / "ckpt-failed-writer"

    result = run_gloo_subprocess_group(
        world_size=world_size,
        fault_plan=None,
        bounds=_BOUNDS,
        tmp_path=tmp_path,
        worker_module=WORKER_MODULE,
        worker_extra_args=[
            "--scenario", "publication", "--checkpoint-root", str(root), "--failure-rank", "1",
        ],
    )

    # The failure must be visible in the process exit codes, separately from any
    # scheduler or harness status.
    assert result.exit_codes[1] not in (0, None), (
        f"rank 1 was supposed to die during save; exit_codes={result.exit_codes}"
    )
    assert result.all_reaped, "the harness must leave no survivor even on this path"

    published = root / "generations"
    selectable = []
    if published.exists():
        selectable = [
            p.name for p in published.iterdir() if (p / "COMPLETE").exists()
        ]
    assert selectable == [], (
        f"a failed shard writer left a selectable generation: {selectable}"
    )
    # And the marker must not be lurking in staging either, since a reader that
    # globbed one level up would then find it.
    staged_complete = list((root / "staging").rglob("COMPLETE")) if (root / "staging").exists() else []
    assert staged_complete == [], f"a COMPLETE marker survives in staging: {staged_complete}"


# --- A-G6 -------------------------------------------------------------------


def test_a_g6_reduction_count_does_not_grow_with_sampling(tmp_path: Path) -> None:
    """A-G6: gradient reductions per update do not grow with MCMC proposal count.

    MEASURED at two different step counts, not asserted by inspection. The count
    is also required to be NONZERO: a counter reading zero under both workloads
    would satisfy "does not grow" vacuously, which is exactly what this gate is
    for.
    """

    _require_capabilities()
    world_size = 2

    def reductions_at(mcmc_steps: int) -> list[dict]:
        result = run_gloo_subprocess_group(
            world_size=world_size,
            fault_plan=None,
            bounds=_BOUNDS,
            tmp_path=tmp_path,
            worker_module=WORKER_MODULE,
            worker_extra_args=["--scenario", "comm-count", "--mcmc-steps", str(mcmc_steps)],
        )
        invocation = Path(result.invocation_dir)
        states = [
            json.loads((invocation / f"state_{rank}.json").read_text())
            for rank in range(world_size)
        ]
        for rank, state in enumerate(states):
            assert state["ok"], f"rank {rank} failed: {state.get('failure_message')}"
        return [state["result"] for state in states]

    short = reductions_at(1)
    long = reductions_at(5)

    for rank in range(world_size):
        # Sampling alone performs NO gradient collectives.
        assert short[rank]["reductions_after_sampling_only"] == 0
        assert long[rank]["reductions_after_sampling_only"] == 0
        # The raw-module work actually happened, so the zero above is meaningful.
        assert long[rank]["raw_forwards"] > short[rank]["raw_forwards"], (
            "the long workload must genuinely do more sampling work"
        )
        # Non-vacuity: the counter must be able to count.
        assert short[rank]["reductions_after_one_update"] > 0, (
            "zero reductions under both workloads would satisfy this gate vacuously"
        )
        assert (
            short[rank]["reductions_after_one_update"]
            == long[rank]["reductions_after_one_update"]
        ), (
            f"rank {rank}: reductions grew with MCMC steps "
            f"({short[rank]['reductions_after_one_update']} -> "
            f"{long[rank]['reductions_after_one_update']})"
        )
        assert long[rank]["prepared_forwards"] == 1, (
            "exactly one prepared forward per update, regardless of sampling work"
        )
