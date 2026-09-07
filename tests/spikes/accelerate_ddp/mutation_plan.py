"""Named source mutations proving each DS-A0 gate is capable of failing.

WHY THIS EXISTS. The acceptance contract requires every gate demonstrated RED
before the code that makes it GREEN. These gates were written after the code, a
deviation disclosed in the commit that introduced them. This module closes it the
way DS-N0 did: for each gate, a NAMED mutation of the source, the gate run
against the mutant expecting failure, the source restored byte-identically, and
the gate re-run expecting success.

That is weaker than genuine test-first in one specific way, and the receipt must
say so: test-first also proves the test was not shaped to fit code that already
existed. A mutation arm proves only that the gate DETECTS the named defect. It is
the stronger half of what test-first buys, not the whole of it.

EACH MUTATION MUST BREAK BEHAVIOUR, NOT MERELY REPORTING. A mutation that only
changes what a worker writes into its state artifact would prove the assertion
reads that field, not that the gate would catch a real defect. Where a mutation
does target reporting, it is marked so a reader is not misled.

THE ANCHOR MUST MATCH EXACTLY ONCE. A mutation that silently fails to apply, or
applies in two places, produces a false green or an unattributable red. The
driver enforces the count and refuses rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Test node ids, spelled exactly as pytest reports them. A parametrize label is
# an INTERFACE: renaming an id changes behaviour not at all and identity
# completely, so a stale id here silently selects nothing and the arm reports a
# vacuous pass.
NODE_A_G0 = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g0_accelerate_capability_is_present_and_recorded"
NODE_A_G1_W2 = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g1_matches_concatenated_oracle[world2-regular]"
NODE_A_G1_W3 = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g1_matches_concatenated_oracle[world3-m2-uneven]"
NODE_A_G1B_CENTERING = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g1b_local_centering_disagrees_with_the_global_oracle"
NODE_A_G1B_CLIPPING = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g1b_per_shard_clipping_disagrees_with_global_clipping"
NODE_A_G2 = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g2_global_zero_finite_count_refuses_before_backward"
NODE_A_G4_SKIP = "tests/unit/training/test_ds_a_accelerate_spike.py::test_a_g4_collective_failure_is_bounded_and_attributed[skip-collective]"


@dataclass(frozen=True)
class Mutation:
    """One named source mutation and the gate it must make fail.

    Parameters
    ----------
    name : str
        Stable identifier for the arm, reported in the driver summary.
    path : str
        Repository-relative file the mutation edits.
    anchor : str
        Exact source text to replace. Must occur EXACTLY ONCE in ``path``.
    replacement : str
        Text to substitute for ``anchor``.
    node_id : str
        The single pytest node id expected to go from green to red.
    defect : str
        The real-world defect this mutant impersonates. This is the sentence
        that says what the arm actually buys.
    kind : str
        ``source`` for a mutation of repository bytes; ``environment`` for one
        that changes the environment instead. The two carry DIFFERENT
        restoration guarantees and must not be read as one class.
    """

    name: str
    path: str
    anchor: str
    replacement: str
    node_id: str
    defect: str
    kind: str = "source"


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="A-G1/drop-world-size-compensation",
        path="tests/spikes/accelerate_ddp/vmc_step.py",
        anchor="    scale = 2.0 * world_size / stats.finite_count",
        replacement="    scale = 2.0 / stats.finite_count",
        node_id=NODE_A_G1_W2,
        defect=(
            "The reducer AVERAGES gradients across ranks, so a surrogate without the "
            "W factor yields gradients a factor of W too small. This is the single "
            "easiest averaging-convention error to make and it produces a plausible, "
            "silently wrong result rather than any error."
        ),
    ),
    Mutation(
        name="A-G1/center-on-rank-local-mean",
        path="tests/spikes/accelerate_ddp/statistics.py",
        anchor="    packets = runtime.all_gather_objects(FiniteStatistics.from_values(values).as_dict())",
        replacement="    packets = [FiniteStatistics.from_values(values).as_dict()]",
        node_id=NODE_A_G1_W3,
        defect=(
            "Skipping the cross-rank gather makes every rank center on its own mean "
            "and divide by its own finite count. On the M2 fixture one shard is "
            "entirely nonfinite, so this also exercises the degenerate path. This is "
            "the defect the whole concatenated-oracle contract exists to catch."
        ),
    ),
    Mutation(
        name="A-G1/publish-the-surrogate-as-the-loss",
        path="tests/spikes/accelerate_ddp/vmc_step.py",
        anchor="        global_loss=global_loss,",
        replacement="        global_loss=float(local_surrogate.detach().item()),",
        node_id=NODE_A_G1_W2,
        defect=(
            "Publishing the world-size-compensated backward surrogate instead of the "
            "global scientific loss. The contract forbids this explicitly; the two "
            "differ by exactly the factor W, so a reader comparing runs at one world "
            "size would never notice."
        ),
    ),
    Mutation(
        name="A-G1b/make-the-wrong-control-right",
        path="tests/spikes/accelerate_ddp/statistics.py",
        anchor="        local_mean = finite_energy.detach().mean()",
        replacement="        local_mean = torch.tensor(0.0, dtype=torch.float64)",
        node_id=NODE_A_G1B_CENTERING,
        defect=(
            "Neutering the negative control so it no longer centers locally. If the "
            "control stops being wrong, A-G1b passes while demonstrating nothing -- "
            "the exact vacuity a negative control exists to prevent."
        ),
    ),
    Mutation(
        name="A-G1b/per-shard-clip-becomes-global-clip",
        path="tests/helpers/vmc_scientific_oracle.py",
        anchor="        total = total + clip_by_global_norm(gradient, clip_norm)",
        replacement="        total = total + gradient",
        node_id=NODE_A_G1B_CLIPPING,
        defect=(
            "Making the naive per-shard clip agree with the global clip, so the "
            "separating fixture no longer separates. NOTE: this mutant edits SHARED "
            "DF0 substrate rather than spike-local code, because that is where the "
            "wrong-comparison-point lives. It is restored byte-identically and the "
            "restoration is verified by sha256 like every other arm."
        ),
    ),
    Mutation(
        name="A-G4/break-the-culprit-self-report",
        path="tests/spikes/accelerate_ddp/worker.py",
        anchor='        f"ddp harness injected fault: rank {rank} phase {phase_name} kind {plan.kind.name}",',
        replacement='        f"fault applied on rank {rank} at {phase_name} ({plan.kind.name})",',
        node_id=NODE_A_G4_SKIP,
        defect=(
            "Reformatting the culprit self-report. The harness derives "
            "culprit_rank by matching this EXACT line, so a reformat silently "
            "destroys attribution while the run still fails and every other "
            "assertion still passes. Chosen deliberately on the skip-collective "
            "path, where the culprit exits CLEANLY and an innocent peer dies on "
            "timeout -- so exit codes alone would finger the wrong rank and the "
            "self-report is the only thing standing between the gate and a "
            "confident wrong answer."
        ),
    ),
    Mutation(
        name="A-G2/forward-before-the-refusal",
        path="tests/spikes/accelerate_ddp/worker.py",
        anchor="    refused = stats.finite_count == 0",
        replacement="    access.score_forward(features)\n    refused = stats.finite_count == 0",
        node_id=NODE_A_G2,
        defect=(
            "Running a prepared forward before the global-M==0 refusal. This is a "
            "genuine behaviour change, not a reporting change: the refusal is "
            "supposed to precede any wrapped forward, and a forward here would enlist "
            "the step in gradient synchronization on a batch that must not produce a "
            "gradient at all."
        ),
    ),
)


# A-G0's red arm has no source mutant and must not be given a fake one.
#
# A-G0 asserts that the Accelerate capability was PRESENT. The only honest way to
# make it fail is to remove the capability, which is an ENVIRONMENT mutation: a
# second `UV_PROJECT_ENVIRONMENT` synced without the `accelerate` extra. It
# therefore cannot ride the revert-byte-identically discipline the source mutants
# use, and it must never appear in a mutant table as though it shares that
# provenance and restoration guarantee.
#
# OBSERVED RED on Cannon job 44954901: torch imported in the same interpreter --
# the control proving the interpreter was not simply broken -- and `import
# accelerate` then raised ModuleNotFoundError.
ENVIRONMENT_ARMS = (
    {
        "name": "A-G0/capability-absent",
        "node_id": NODE_A_G0,
        "kind": "environment",
        "how": "sync a separate UV_PROJECT_ENVIRONMENT with --extra cpu only",
        "observed_red_in": "Cannon job 44954901",
        "defect": (
            "A suite running where the optional extra is absent would skip every "
            "A-G* arm and report green. A-G0 asserts rather than skips so that "
            "environment fails loudly instead of silently."
        ),
    },
)


__all__ = ["ENVIRONMENT_ARMS", "MUTATIONS", "Mutation"]
