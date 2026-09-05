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
    reviewed_tests=("test_r2_reviewed_n_g4_fault_tests_preserve_evidence",),
    oracle_test="test_r2_reviewer_n_g4_broad_exit_fact_oracle",
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
    MutationArm(13, "test_r2_n_g5_digest_change_blocks_publication", "tests/spikes/native_ddp/checkpoint.py", "                if payload != actual.as_dict():\n", "                if False:\n", "digest changed"),
    MutationArm(14, "test_r2_n_g5_delayed_writer_does_not_publish_partial_generation", "tests/spikes/native_ddp/checkpoint.py", "            time.sleep(delay_seconds)\n", "            time.sleep(0.0)\n", "R2-G5 delayed writer blocks publication"),
    MutationArm(15, "test_r2_n_g6_reduction_count_does_not_scale_with_sampling_work", "tests/spikes/native_ddp/vmc_step.py", "    access.ddp_model.register_comm_hook(counter, counter.communication_hook)\n", "", "R2-G6 update reduction count"),
    MutationArm(16, "test_r2_n_e2_coordinate_work_uses_raw_model_only", "tests/spikes/native_ddp/model_access.py", "        logabs = self.raw_model(coordinates)\n", "        logabs = self.ddp_model(coordinates)\n", "coordinate work must not use DDP wrapper"),
    MutationArm(17, "test_r2_n_e3_sgd_and_adam_have_nonempty_independent_optimizer_evidence", "tests/spikes/native_ddp/worker.py", "        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n", "        return torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)\n", "R2-E3 sgd complete optimizer state matches independent reference"),
    MutationArm(18, "test_r2_n_e3_optimizer_state_and_closure_objective_are_global", "tests/spikes/native_ddp/vmc_step.py", "        return torch.tensor(global_loss, dtype=local_surrogate.dtype)\n", "        return local_surrogate\n", "closure parameters must be globally synchronized"),
    MutationArm(19, "test_r2_n_e4_inventory_names_consumed_dcp_apis_and_classifies_them", "tests/spikes/native_ddp/worker.py", "                \"torch.distributed.checkpoint.state_dict.get_state_dict\",\n", "                \"torch.distributed.checkpoint.state_dict.get_state_dict_broken\",\n", "R2-ARM-19 consumed DCP API inventory"),
    MutationArm(20, "test_r2_n_e5_state_and_receipt_are_rank_attributed", "tests/spikes/native_ddp/worker.py", "        \"hostname\": os.uname().nodename,\n        \"pid\": os.getpid(),\n        \"access\": {\n", "        \"hostname\": os.uname().nodename,\n        \"access\": {\n", "pid"),
    MutationArm(21, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/helpers/ddp_subprocess_harness.py", "            worker_module,\n", "            \"tests.helpers.ddp_worker_entrypoint\",\n", "R2-ARM-21 worker entrypoint publication"),
)


def apply_once(path: Path, old: str, new: str) -> bytes:
    original = path.read_bytes()
    count = original.count(old.encode())
    if count != 1:
        raise ValueError(f"expected one exact anchor in {path}, found {count}")
    path.write_bytes(original.replace(old.encode(), new.encode(), 1))
    return original


def _run(root: Path, python: str, test: str) -> subprocess.CompletedProcess[str]:
    selector = test if "::" in test else f"tests/unit/training/test_ds_n_r2_review.py::{test}"
    return subprocess.run(
        [python, "-m", "pytest", "-q", selector],
        cwd=root, text=True, capture_output=True, check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + "\n" + result.stderr


_BROKEN = re.compile(
    r"(?:SyntaxError|IndentationError|ImportError|ModuleNotFoundError|NameError|"
    r"INTERNALERROR|command not found|No such file|pytest: error)"
)


def _assert_green(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label}: expected green\n{_output(result)}")


def _assert_semantic_red(result: subprocess.CompletedProcess[str], arm: MutationArm) -> None:
    output = _output(result)
    if result.returncode == 0:
        raise RuntimeError(f"arm {arm.number}: expected semantic red, got green")
    if _BROKEN.search(output):
        raise RuntimeError(f"arm {arm.number}: broken mutation/setup failure\n{output}")
    if any(token in output for token in ("TypeError", "AttributeError")):
        raise RuntimeError(f"arm {arm.number}: TypeError/AttributeError is not semantic evidence\n{output}")
    if arm.expected not in output:
        raise RuntimeError(f"arm {arm.number}: missing semantic signature {arm.expected!r}\n{output}")


def _assert_composite_semantic_red(
    result: subprocess.CompletedProcess[str], mutation: CompositeMutation
) -> None:
    output = _output(result)
    if result.returncode == 0:
        raise RuntimeError(f"{mutation.name}: expected semantic red, got green")
    if _BROKEN.search(output) or any(token in output for token in ("TypeError", "AttributeError")):
        raise RuntimeError(f"{mutation.name}: broken mutation/setup failure\n{output}")
    if mutation.expected not in output:
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
        f"RED_RC={red.returncode} SIGNATURE={arm.expected!r} RESTORED_GREEN=1",
        flush=True,
    )
    print(f"ARM={arm.number} MUTATED_OUTPUT_BEGIN\n{_output(red)}ARM={arm.number} MUTATED_OUTPUT_END", flush=True)
    return 0


def run_composite(root: Path, mutation: CompositeMutation, *, python: str) -> int:
    for test in (*mutation.reviewed_tests, mutation.oracle_test):
        _assert_green(_run(root, python, test), f"{mutation.name} baseline {test}")
    originals: dict[Path, bytes] = {}
    mutated_digests: list[str] = []
    try:
        for relative_path, old, new in mutation.replacements:
            path = root / relative_path
            if path not in originals:
                originals[path] = path.read_bytes()
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
        f"SIGNATURE={mutation.expected!r} RESTORED_GREEN=1",
        flush=True,
    )
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
