"""Reviewer-owned, exact-once DS-N mutation specification and driver."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MutationArm:
    number: int
    test: str
    relative_path: str
    old: str
    new: str
    expected: str
    reviewed_test: str | None = None


@dataclass(frozen=True)
class CompositeMutation:
    name: str
    replacements: tuple[tuple[str, str, str], ...]
    reviewed_tests: tuple[str, ...]
    oracle_test: str
    expected: str


CLEAN_ALL_RANK_EARLY_ABORT = CompositeMutation(
    name="clean-all-rank-early-abort",
    replacements=(
        (
            "tests/spikes/native_ddp/worker.py",
            "        raise RuntimeError(f\"ddp harness injected fault: rank {rank} phase {phase.name}\")\n",
            "        if False:\n            raise RuntimeError(f\"ddp harness injected fault: rank {rank} phase {phase.name}\")\n",
        ),
        (
            "tests/spikes/native_ddp/worker.py",
            "        reductions_before = counter.count\n",
            "        if plan is not None and plan.kind == FaultKind.RAISE_BEFORE_BACKWARD:\n"
            "            state = _common_observability(rank=args.rank, world_size=args.world_size, access=access, optimizer=optimizer)\n"
            "            state.update({\"status\": \"early_aborted\", \"phase_sequence\": phase_sequence, \"fault_kind\": plan.kind.name, \"fault_applied_rank\": plan.target_rank, \"fault_phase\": plan.phase.name})\n"
            "            Path(args.state_path).write_text(json.dumps(state, sort_keys=True))\n"
            "            runtime.barrier()\n"
            "            runtime.close()\n"
            "            return 0\n"
            "        reductions_before = counter.count\n",
        ),
    ),
    reviewed_tests=(
        "test_r2_reviewed_n_g4_fault_tests_preserve_evidence",
        "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_raise_before_backward_preserves_culprit_and_nonpublication",
        "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_skip_collective_preserves_culprit_and_nonpublication",
        "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_stall_before_collective_is_bounded_and_nonpublishing",
    ),
    oracle_test="test_r2_reviewer_n_g4_broad_exit_fact_oracle_raise",
    expected="R2-G4 broad exit fact: at least one nonzero child exit",
)


ARMS = (
    MutationArm(1, "test_r2_n_g1_world_size_compensates_the_ddp_surrogate", "tests/spikes/native_ddp/vmc_step.py", "scale = 2.0 * world_size / stats.finite_count", "scale = 2.0 / stats.finite_count", "R2-ARM-01 observed ddp_backward_scale"),
    MutationArm(2, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/statistics.py", "if other.finite_count == 0:", "if other.finite_count != 0:", "R2-ARM-02 global finite count"),
    MutationArm(3, "test_r2_n_g1b_rejects_local_centering_against_global_oracle", "tests/spikes/native_ddp/statistics.py", "result = result + ((finite_energy.detach() - local_mean) * logabs[mask]).sum()", "result = result + (finite_energy.detach() * logabs[mask]).sum()", "R2-ARM-03 global oracle mismatch"),
    MutationArm(4, "test_r2_n_g2_all_invalid_has_no_backward_or_parameter_gradient_event", "tests/spikes/native_ddp/worker.py", "        if stats.finite_count == 0:\n", "        raw_features = last_coordinates.detach().clone().requires_grad_(True)\n        access.raw_model(raw_features).sum().backward()\n        optimizer.zero_grad(set_to_none=True)\n        if stats.finite_count == 0:\n", "R2-ARM-04 parameter-gradient event", "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_global_zero_valid_energy_refuses_before_backward_and_optimizer_mutation"),
    MutationArm(5, "test_r2_n_g3_resume_applies_detached_dcp_buffers", "tests/spikes/native_ddp/checkpoint.py", "        set_state_dict(\n            model,\n            optimizer,\n            model_state_dict=model_state,\n            optim_state_dict=optimizer_state,\n        )\n", "", "R2-ARM-05 resumed parameters"),
    MutationArm(6, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "        if False:\n", "R2-ARM-06 DCP load boundary entered"),
    MutationArm(7, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "        with torch.no_grad():\n            model.weight.add_(1.0)\n        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "R2-ARM-07 model unchanged"),
    MutationArm(8, "test_r2_n_g3b_perturbed_rank_sidecar_is_rejected", "tests/spikes/native_ddp/checkpoint.py", "            if actual.as_dict() != expected:\n", "            if False:\n", "R2-ARM-08 sidecar validation failure"),
    MutationArm(9, "test_r2_reviewed_n_g4_fault_tests_preserve_evidence", "tests/spikes/native_ddp/worker.py", "        raise RuntimeError(f\"ddp harness injected fault: rank {rank} phase {phase.name}\")\n", "        return\n", "R2-G4 no publication after configured fault"),
    MutationArm(10, "test_r2_reviewed_n_g4_fault_tests_preserve_evidence", "tests/spikes/native_ddp/worker.py", "        os._exit(2)\n", "        return\n", "R2-G4 no publication after configured fault"),
    MutationArm(11, "test_r2_reviewed_n_g4_fault_tests_preserve_evidence", "tests/spikes/native_ddp/worker.py", "        time.sleep(plan.delay_seconds)\n", "        time.sleep(0.0)\n", "R2-G4 no publication after configured fault"),
    MutationArm(12, "test_r2_n_g5_failed_publication_keeps_previous_generation_selectable", "tests/spikes/native_ddp/worker.py", "        state[\"status\"] = \"checkpoint_pending\"\n", "        state[\"status\"] = \"success\"\n", "R2-G5 failed publication must retain checkpoint_pending state"),
    MutationArm(13, "test_r2_n_g5_digest_change_blocks_publication", "tests/spikes/native_ddp/checkpoint.py", "                if payload != actual.as_dict():\n", "                if False:\n", "R2-ARM-13 digest guard did not raise"),
    MutationArm(14, "test_r2_n_g5_delayed_writer_does_not_publish_partial_generation", "tests/spikes/native_ddp/checkpoint.py", "            time.sleep(delay_seconds)\n", "            time.sleep(0.0)\n", "R2-G5 delayed writer blocks publication"),
    MutationArm(15, "test_r2_n_g6_reduction_count_does_not_scale_with_sampling_work", "tests/spikes/native_ddp/vmc_step.py", "    access.ddp_model.register_comm_hook(counter, counter.communication_hook)\n", "", "R2-G6 update reduction count"),
    MutationArm(16, "test_r2_n_e2_coordinate_work_uses_raw_model_only", "tests/spikes/native_ddp/model_access.py", "        logabs = self.raw_model(coordinates)\n", "        logabs = self.ddp_model(coordinates)\n", "R2-ARM-16 coordinate work must not use DDP wrapper"),
    MutationArm(17, "test_r2_n_e3_sgd_and_adam_have_nonempty_independent_optimizer_evidence", "tests/spikes/native_ddp/worker.py", "        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n", "        return torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)\n", "R2-ARM-17 parameters match independent reference"),
    MutationArm(18, "test_r2_n_e3_optimizer_state_and_closure_objective_are_global", "tests/spikes/native_ddp/vmc_step.py", "        return torch.tensor(global_loss, dtype=local_surrogate.dtype)\n", "        return local_surrogate\n", "R2-ARM-18 closure parameters must be globally synchronized"),
    MutationArm(19, "test_r2_n_e4_inventory_names_consumed_dcp_apis_and_classifies_them", "tests/spikes/native_ddp/worker.py", "                \"torch.distributed.checkpoint.state_dict.get_state_dict\",\n", "                \"torch.distributed.checkpoint.state_dict.get_state_dict_broken\",\n", "R2-ARM-19 consumed DCP API inventory"),
    MutationArm(20, "test_r2_n_e5_state_and_receipt_are_rank_attributed", "tests/spikes/native_ddp/worker.py", "        \"hostname\": os.uname().nodename,\n        \"pid\": os.getpid(),\n        \"access\": {\n", "        \"hostname\": os.uname().nodename,\n        \"access\": {\n", "R2-ARM-20 pid field"),
    MutationArm(21, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/helpers/ddp_subprocess_harness.py", "            worker_module,\n", "            \"tests.helpers.ddp_worker_entrypoint\",\n", "R2-ARM-21 worker entrypoint publication"),
    # Supplemental probes preserve the original lane-4 and lane-6 mutations;
    # reviewer arms 4 and 7 remain the independent raw/guard probes.
    MutationArm(22, "test_r2_n_g2_all_invalid_has_no_backward_or_parameter_gradient_event", "tests/spikes/native_ddp/worker.py", "        if stats.finite_count == 0:\n", "        ddp_features = last_coordinates.detach().clone().requires_grad_(True)\n        access.score_forward(ddp_features).sum().backward()\n        optimizer.zero_grad(set_to_none=True)\n        if stats.finite_count == 0:\n", "R2-ARM-04 parameter-gradient event", "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_global_zero_valid_energy_refuses_before_backward_and_optimizer_mutation"),
    MutationArm(23, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "        with torch.no_grad():\n            model.weight.add_(1.0)\n        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "R2-ARM-07 model unchanged", "tests/unit/training/test_ds_n_native_ddp_spike.py::test_native_topology_change_is_refused_before_any_resume_mutation"),
)


LANE_ARM_MAPPING = {
    1: "reviewer arm 1 scale oracle",
    2: "reviewer arm 2 M2 statistics oracle",
    3: "reviewer arm 3 global-centering oracle",
    4: "reviewer arm 4 raw-model differential; supplemental arm 22 original lane DDP backward",
    5: "reviewer arm 5 detached DCP restore",
    6: "reviewer arm 6 topology-guard boundary; supplemental arm 23 integrated lane topology refusal",
    7: "reviewer arm 7 pre-guard model invariance",
    8: "reviewer arm 8 sampler-shard digest validation",
    9: "reviewer arm 9 raise fault publication gate",
    10: "reviewer arm 10 skip/crash fault publication gate",
    11: "reviewer arm 11 stall fault publication gate",
    12: "reviewer arm 12 failed publication state",
    13: "reviewer arm 13 digest guard",
    14: "reviewer arm 14 delayed publication",
    15: "reviewer arm 15 reducer count",
    16: "reviewer arm 16 raw coordinate ownership",
    17: "reviewer arm 17 SGD/Adam reference",
    18: "reviewer arm 18 global closure state",
    19: "reviewer arm 19 DCP inventory",
    20: "reviewer arm 20 rank PID attribution",
    21: "reviewer arm 21 worker entrypoint selection",
}


def apply_once(path: Path, old: str, new: str) -> bytes:
    original = path.read_bytes()
    count = original.count(old.encode())
    if count != 1:
        raise ValueError(f"expected one exact anchor in {path}, found {count}")
    path.write_bytes(original.replace(old.encode(), new.encode(), 1))
    return original


def _run(root: Path, python: str, test: str) -> subprocess.CompletedProcess[str]:
    selector = test if "::" in test else f"tests/unit/training/test_ds_n_r2_review.py::{test}"
    verbosity = ["-s"] if "skip_collective_broad_exit" in test else []
    return subprocess.run(
        [python, "-m", "pytest", "-q", *verbosity, selector],
        cwd=root, text=True, capture_output=True, check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + "\n" + result.stderr


_BROKEN = re.compile(
    r"(?:SyntaxError|IndentationError|ImportError|ModuleNotFoundError|NameError|"
    r"INTERNALERROR|command not found|No such file|pytest: error)"
)


def _assert_green(result: subprocess.CompletedProcess[str], label: str) -> None:
    output = _output(result)
    if result.returncode != 0 or re.search(r"(?:no tests ran|collected 0 items)", output, re.IGNORECASE):
        raise RuntimeError(f"{label}: expected green\n{_output(result)}")


def _assert_semantic_red(result: subprocess.CompletedProcess[str], arm: MutationArm) -> None:
    output = _output(result)
    if result.returncode == 0:
        raise RuntimeError(f"arm {arm.number}: expected semantic red, got green")
    if _BROKEN.search(output):
        raise RuntimeError(f"arm {arm.number}: broken mutation/setup failure\n{output}")
    if any(token in output for token in ("TypeError", "AttributeError")):
        raise RuntimeError(f"arm {arm.number}: TypeError/AttributeError is not semantic evidence\n{output}")
    diagnostic = re.compile(
        rf"^E\s+(?:AssertionError|Failed):.*{re.escape(arm.expected)}",
        re.MULTILINE,
    )
    if not diagnostic.search(output):
        raise RuntimeError(f"arm {arm.number}: missing semantic signature {arm.expected!r}\n{output}")


def _assert_composite_semantic_red(
    result: subprocess.CompletedProcess[str], mutation: CompositeMutation
) -> None:
    output = _output(result)
    if result.returncode == 0:
        raise RuntimeError(f"{mutation.name}: expected semantic red, got green")
    if _BROKEN.search(output) or any(token in output for token in ("TypeError", "AttributeError")):
        raise RuntimeError(f"{mutation.name}: broken mutation/setup failure\n{output}")
    diagnostic = re.compile(
        rf"^E\s+(?:AssertionError|Failed):.*{re.escape(mutation.expected)}",
        re.MULTILINE,
    )
    if not diagnostic.search(output):
        raise RuntimeError(f"{mutation.name}: missing semantic signature {mutation.expected!r}\n{output}")


def run_arm(root: Path, arm: MutationArm, *, python: str) -> int:
    _assert_green(_run(root, python, arm.test), f"arm {arm.number} baseline")
    if arm.reviewed_test:
        _assert_green(_run(root, python, arm.reviewed_test), f"arm {arm.number} reviewed baseline")
    path = root / arm.relative_path
    original = path.read_bytes()
    original_digest = hashlib.sha256(original).hexdigest()
    mutated_digest: str | None = None
    red: subprocess.CompletedProcess[str] | None = None
    try:
        mutated_original = apply_once(path, arm.old, arm.new)
        mutated_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if mutated_original != original or mutated_digest == original_digest:
            raise RuntimeError(f"arm {arm.number}: mutation digest proof failed")
        if arm.reviewed_test:
            _assert_green(_run(root, python, arm.reviewed_test), f"arm {arm.number} reviewed differential")
        red = _run(root, python, arm.test)
        _assert_semantic_red(red, arm)
    finally:
        path.write_bytes(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != original_digest:
            raise RuntimeError(f"arm {arm.number}: byte-identical restore failed")
        _assert_green(_run(root, python, arm.test), f"arm {arm.number} post-restore")
        if arm.reviewed_test:
            _assert_green(_run(root, python, arm.reviewed_test), f"arm {arm.number} reviewed post-restore")
    if red is None or mutated_digest is None:
        raise RuntimeError(f"arm {arm.number}: no complete mutation/red proof")
    print(
        f"ARM={arm.number} BASELINE_RC=0 MUTATED_DIGEST={mutated_digest} "
        f"RED_RC={red.returncode} SIGNATURE={arm.expected!r} "
        f"ORIGINAL_DIGEST={original_digest} RESTORED_DIGEST={original_digest} RESTORED_GREEN=1",
        flush=True,
    )
    print(f"ARM={arm.number} MUTATED_OUTPUT_BEGIN\n{_output(red)}ARM={arm.number} MUTATED_OUTPUT_END", flush=True)
    return 0


def run_composite(root: Path, mutation: CompositeMutation, *, python: str) -> int:
    print("LANE_ARM_MAPPING_BEGIN", flush=True)
    for number in range(1, 22):
        print(f"LANE_ARM_{number}={LANE_ARM_MAPPING[number]}", flush=True)
    print("LANE_ARM_MAPPING_END", flush=True)
    for test in (*mutation.reviewed_tests, mutation.oracle_test):
        baseline = _run(root, python, test)
        _assert_green(baseline, f"{mutation.name} baseline {test}")
        if "skip_collective_broad_exit" in test:
            print(f"{mutation.name} SKIP_BASELINE_OUTPUT_BEGIN\n{_output(baseline)}{mutation.name} SKIP_BASELINE_OUTPUT_END", flush=True)
    originals: dict[Path, bytes] = {}
    original_digests: dict[Path, str] = {}
    mutated_digests: list[str] = []
    try:
        for relative_path, old, new in mutation.replacements:
            path = root / relative_path
            if path not in originals:
                originals[path] = path.read_bytes()
                original_digests[path] = hashlib.sha256(originals[path]).hexdigest()
            apply_once(path, old, new)
            mutated_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        for test in mutation.reviewed_tests:
            _assert_green(_run(root, python, test), f"{mutation.name} reviewed differential")
        red = _run(root, python, mutation.oracle_test)
        _assert_composite_semantic_red(red, mutation)
        red_rc = red.returncode
    finally:
        for path, original in originals.items():
            path.write_bytes(original)
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest():
                raise RuntimeError(f"{mutation.name}: byte-identical restore failed")
        for test in (*mutation.reviewed_tests, mutation.oracle_test):
            _assert_green(_run(root, python, test), f"{mutation.name} post-restore {test}")
    print(
        f"COMPOSITE={mutation.name} BASELINE_GREEN=1 REVIEWED_GREEN=1 "
        f"MUTATED_DIGESTS={','.join(mutated_digests)} RED_RC={red_rc} "
        f"ORIGINAL_DIGESTS={','.join(original_digests.values())} "
        f"RESTORED_DIGESTS={','.join(original_digests.values())} "
        f"SIGNATURE={mutation.expected!r} RESTORED_GREEN=1",
        flush=True,
    )
    print(f"COMPOSITE={mutation.name} MUTATED_OUTPUT_BEGIN\n{_output(red)}COMPOSITE={mutation.name} MUTATED_OUTPUT_END", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", type=int)
    parser.add_argument("--clean-early-abort", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.clean_early_abort:
        return run_composite(args.root, CLEAN_ALL_RANK_EARLY_ABORT, python=args.python)
    if args.arm is None:
        parser.error("one of --arm or --clean-early-abort is required")
    arm = next(item for item in ARMS if item.number == args.arm)
    return run_arm(args.root, arm, python=args.python)


if __name__ == "__main__":
    raise SystemExit(main())
