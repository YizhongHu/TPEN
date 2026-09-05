"""Reviewer-owned mutation specification and reversible execution driver.

This driver deliberately operates on the checkout supplied by the caller.  It
does not import or invoke any lane driver, job, or output.  Every mutation is
anchored by exact old bytes and is restored in a ``finally`` block.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
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


@dataclass(frozen=True)
class CompositeMutation:
    """A single semantic mutation consisting of coordinated exact edits."""

    name: str
    replacements: tuple[tuple[str, str, str], ...]
    reviewed_tests: tuple[str, ...]
    oracle_test: str


CLEAN_ALL_RANK_EARLY_ABORT = CompositeMutation(
    name="clean-all-rank-early-abort",
    replacements=(
        (
            "tests/spikes/native_ddp/worker.py",
            "        os._exit(2)\n",
            "        return\n",
        ),
        (
            "tests/spikes/native_ddp/worker.py",
            "        stats = prepare_statistics(runtime, last_energy)\n",
            "        if plan is not None and plan.kind == FaultKind.SKIP_COLLECTIVE:\n"
            "            state = _common_observability(rank=args.rank, world_size=args.world_size, access=access, optimizer=optimizer)\n"
            "            state.update({\"status\": \"early_aborted\", \"phase_sequence\": phase_sequence, \"fault_kind\": plan.kind.name})\n"
            "            Path(args.state_path).write_text(json.dumps(state, sort_keys=True))\n"
            "            runtime.barrier()\n"
            "            runtime.close()\n"
            "            _write_receipt(args, phase_sequence, plan.kind.name)\n"
            "            return 0\n"
            "        stats = prepare_statistics(runtime, last_energy)\n",
        ),
    ),
    reviewed_tests=(
        "test_r2_reviewed_n_g4_fault_tests_preserve_evidence",
    ),
    oracle_test="test_r2_reviewer_n_g4_broad_exit_fact_oracle",
)


ARMS = (
    MutationArm(1, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/vmc_step.py", "scale = 2.0 * world_size / stats.finite_count", "scale = 2.0 / stats.finite_count", "finite_count mismatch or surrogate scale"),
    MutationArm(2, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/statistics.py", "if other.finite_count == 0:", "if other.finite_count != 0:", "finite_count mismatch"),
    MutationArm(3, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/statistics.py", "result = result + ((finite_energy.detach() - local_mean) * logabs[mask]).sum()", "result = result + (finite_energy.detach() * logabs[mask]).sum()", "local-centering control"),
    MutationArm(4, "test_r2_n_g2_all_invalid_has_no_backward_or_parameter_gradient_event", "tests/spikes/native_ddp/worker.py", "        if stats.finite_count == 0:\n", "        optimizer.zero_grad(set_to_none=True)\n        access.score_forward(last_coordinates).sum().backward()\n        optimizer.zero_grad(set_to_none=True)\n        if stats.finite_count == 0:\n", "review backward oracle"),
    MutationArm(5, "test_r2_n_g3_resume_applies_detached_dcp_buffers", "tests/spikes/native_ddp/checkpoint.py", "        set_state_dict(\n            model,\n            optimizer,\n            model_state_dict=model_state,\n            optim_state_dict=optimizer_state,\n        )\n", "", "resume state application"),
    MutationArm(6, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "        if False:\n", "topology refusal bypass"),
    MutationArm(7, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "        with torch.no_grad():\n            model.weight.add_(1.0)\n        if int(metadata[\"world_size\"]) != self.runtime.world_size:\n", "pre-gate mutation"),
    MutationArm(8, "test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation", "tests/spikes/native_ddp/checkpoint.py", "            if actual.as_dict() != expected:\n", "            if False:\n", "sidecar digest bypass"),
    MutationArm(9, "test_r2_n_g4_requires_broad_exit_facts_for_each_fault", "tests/spikes/native_ddp/worker.py", "        raise RuntimeError(f\"ddp harness injected fault: rank {rank} phase {phase.name}\")\n", "        return\n", "raise fault neutralized"),
    MutationArm(10, "test_r2_n_g4_requires_broad_exit_facts_for_each_fault", "tests/spikes/native_ddp/worker.py", "        os._exit(2)\n", "        return\n", "skip fault neutralized"),
    MutationArm(11, "test_r2_n_g4_requires_broad_exit_facts_for_each_fault", "tests/spikes/native_ddp/worker.py", "        time.sleep(plan.delay_seconds)\n", "        time.sleep(0.0)\n", "stall removed"),
    MutationArm(12, "test_r2_n_g5_failed_publication_keeps_previous_generation_selectable", "tests/spikes/native_ddp/worker.py", "        state[\"status\"] = \"checkpoint_pending\"\n", "        state[\"status\"] = \"success\"\n", "premature success"),
    MutationArm(13, "test_r2_n_g5_failed_publication_keeps_previous_generation_selectable", "tests/spikes/native_ddp/checkpoint.py", "                if payload != actual.as_dict():\n", "                if False:\n", "digest validation bypass"),
    MutationArm(14, "test_r2_n_g5_failed_publication_keeps_previous_generation_selectable", "tests/spikes/native_ddp/checkpoint.py", "            time.sleep(delay_seconds)\n", "            time.sleep(0.0)\n", "delay removed"),
    MutationArm(15, "test_r2_n_g6_reduction_count_does_not_scale_with_sampling_work", "tests/spikes/native_ddp/vmc_step.py", "    access.ddp_model.register_comm_hook(counter, counter.communication_hook)\n", "", "reduction counter"),
    MutationArm(16, "test_r2_n_e5_state_and_receipt_are_rank_attributed", "tests/spikes/native_ddp/model_access.py", "        logabs = self.raw_model(coordinates)\n", "        logabs = self.ddp_model(coordinates)\n", "raw-model boundary"),
    MutationArm(17, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/worker.py", "        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n", "        return torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)\n", "optimizer trajectory"),
    MutationArm(18, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/spikes/native_ddp/vmc_step.py", "        return torch.tensor(global_loss, dtype=local_surrogate.dtype)\n", "        return local_surrogate\n", "global closure objective"),
    MutationArm(19, "test_r2_n_e4_inventory_names_consumed_dcp_apis_and_classifies_them", "tests/spikes/native_ddp/worker.py", "                \"torch.distributed.checkpoint.state_dict.get_state_dict\",\n", "                \"torch.distributed.checkpoint.state_dict.get_state_dict_broken\",\n", "API inventory"),
    MutationArm(20, "test_r2_n_e5_state_and_receipt_are_rank_attributed", "tests/spikes/native_ddp/worker.py", "        \"hostname\": os.uname().nodename,\n        \"pid\": os.getpid(),\n        \"access\": {\n", "        \"hostname\": os.uname().nodename,\n        \"access\": {\n", "rank pid attribution"),
    MutationArm(21, "test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics", "tests/helpers/ddp_subprocess_harness.py", "            worker_module,\n", "            \"tests.helpers.ddp_worker_entrypoint\",\n", "worker selection"),
)


def apply_once(path: Path, old: str, new: str) -> bytes:
    original = path.read_bytes()
    old_bytes = old.encode()
    if original.count(old_bytes) != 1:
        raise ValueError(f"expected one exact anchor in {path}, found {original.count(old_bytes)}")
    path.write_bytes(original.replace(old_bytes, new.encode(), 1))
    return original


def run_arm(root: Path, arm: MutationArm, *, python: str, pytest_args: list[str]) -> int:
    path = root / arm.relative_path
    original = apply_once(path, arm.old, arm.new)
    before = hashlib.sha256(original).hexdigest()
    try:
        completed = subprocess.run([python, "-m", "pytest", "-q", f"{arm.test}", *pytest_args], cwd=root, check=False)
        return completed.returncode
    finally:
        path.write_bytes(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise RuntimeError(f"restore digest mismatch for arm {arm.number}")


def run_composite(root: Path, mutation: CompositeMutation, *, python: str, pytest_args: list[str]) -> int:
    originals: list[tuple[Path, bytes]] = []
    try:
        for relative_path, old, new in mutation.replacements:
            path = root / relative_path
            originals.append((path, apply_once(path, old, new)))
        reviewed = subprocess.run(
            [python, "-m", "pytest", "-q", "tests/unit/training/test_ds_n_r2_review.py", "-k",
             " or ".join(mutation.reviewed_tests), *pytest_args], cwd=root, check=False
        )
        oracle = subprocess.run(
            [python, "-m", "pytest", "-q", "tests/unit/training/test_ds_n_r2_review.py", "-k",
             mutation.oracle_test, *pytest_args], cwd=root, check=False
        )
        if reviewed.returncode != 0:
            raise RuntimeError(f"{mutation.name}: reviewed tests were not green")
        if oracle.returncode == 0:
            raise RuntimeError(f"{mutation.name}: broad-exit oracle did not go red")
        return oracle.returncode
    finally:
        for path, original in originals:
            path.write_bytes(original)
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest():
                raise RuntimeError(f"restore digest mismatch for {mutation.name}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", type=int)
    parser.add_argument("--clean-early-abort", action="store_true")
    parser.add_argument("--python", default="python")
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args()
    if args.clean_early_abort:
        return run_composite(args.root, CLEAN_ALL_RANK_EARLY_ABORT, python=args.python, pytest_args=args.pytest_args)
    if args.arm is None:
        parser.error("one of --arm or --clean-early-abort is required")
    arm = next(item for item in ARMS if item.number == args.arm)
    return run_arm(args.root, arm, python=args.python, pytest_args=args.pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
