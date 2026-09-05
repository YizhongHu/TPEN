"""Independent, contract-first DS-N round-two evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan
from tests.helpers.ddp_subprocess_harness import HarnessBounds, RankReceipt, run_gloo_subprocess_group
from tests.spikes.native_ddp import checkpoint as checkpoint_module
from tests.spikes.native_ddp.checkpoint import CheckpointPayloadStore, CheckpointTopologyMismatch
from tests.spikes.native_ddp.model_access import SemanticWavefunction
from tests.spikes.native_ddp.statistics import FiniteStatistics, centered_terms, local_centered_objective
from tests.spikes.native_ddp.vmc_step import make_local_surrogate


CAPABILITY = probe_gloo_capability()
FAULT_BOUNDS = HarnessBounds(process_group_timeout=2.0, watchdog_timeout=12.0)
DEFAULT_BOUNDS = HarnessBounds(process_group_timeout=6.0, watchdog_timeout=20.0)


@pytest.fixture(autouse=True)
def require_gloo_subprocess_capability() -> None:
    if not CAPABILITY.gloo_available:
        pytest.skip(missing_capability_reason(CAPABILITY, "gloo_available"))
    if not CAPABILITY.subprocess_spawn_available:
        pytest.skip(missing_capability_reason(CAPABILITY, "subprocess_spawn_available"))


def run_native(tmp_path: Path, *, world_size: int = 2, fault_plan: FaultPlan | None = None,
               bounds: HarnessBounds = DEFAULT_BOUNDS, extra_args: tuple[str, ...] = ()):
    return run_gloo_subprocess_group(
        world_size, fault_plan, bounds, tmp_path,
        worker_module="tests.spikes.native_ddp.worker", worker_extra_args=extra_args,
    )


def run_probe(tmp_path: Path, *, extra_args: tuple[str, ...] = ()):
    return run_gloo_subprocess_group(
        2, None, DEFAULT_BOUNDS, tmp_path,
        worker_module="tests.spikes.native_ddp.review_round2_probe",
        worker_extra_args=extra_args,
    )


def states(result) -> list[dict]:
    return [json.loads((Path(result.invocation_dir) / f"state_{rank}.json").read_text())
            for rank in range(len(result.receipts))]


def assert_success(result) -> None:
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True, "publication_observed"
    assert result.exit_codes == (0,) * len(result.receipts)
    assert all(isinstance(receipt, RankReceipt) for receipt in result.receipts)
    assert all(state["status"] == "success" for state in states(result))


def test_r2_n_g1_m2_observes_uneven_shards_and_global_statistics(tmp_path: Path) -> None:
    result = run_native(tmp_path, world_size=3,
                        extra_args=("--experiment", "scientific", "--fixture", "m2"))
    assert_success(result)
    observed = states(result)
    assert [len(state["energy"]["__tensor__"]) for state in observed] == [5, 3, 7]
    assert [state["global_statistics"]["finite_count"] for state in observed] == [8, 8, 8]
    assert [state["global_statistics"]["total_count"] for state in observed] == [15, 15, 15]


def test_r2_n_g1_world_size_compensates_the_ddp_surrogate(tmp_path: Path) -> None:
    result = run_native(tmp_path, world_size=2, extra_args=("--experiment", "scientific"))
    assert_success(result)
    observed = states(result)
    finite_counts = {state["global_statistics"]["finite_count"] for state in observed}
    assert finite_counts == {6}, "common global finite_count"
    expected_scale = 2.0 * 2 / next(iter(finite_counts))
    assert all(
        state["ddp_backward_scale"] == pytest.approx(expected_scale)
        for state in observed
    ), "ddp_backward_scale must be 2*world_size/M"


def test_r2_n_g1b_rejects_local_centering_against_global_oracle(tmp_path: Path) -> None:
    del tmp_path
    energies = [torch.tensor([1.0]), torch.tensor([5.0])]
    logabs = [torch.tensor([0.3]), torch.tensor([0.7])]
    global_stats = FiniteStatistics.from_values(torch.cat(energies))
    global_oracle = sum(
        centered_terms(log, energy, global_stats).sum()
        for log, energy in zip(logabs, energies, strict=True)
    )
    assert local_centered_objective(logabs, energies).item() == pytest.approx(0.0)
    assert local_centered_objective(logabs, energies).item() != pytest.approx(
        global_oracle.item()
    )


def test_r2_n_g2_all_invalid_has_no_backward_or_parameter_gradient_event(tmp_path: Path) -> None:
    result = run_probe(tmp_path, extra_args=("--experiment", "scientific", "--fixture", "all_invalid"))
    assert result.all_reaped is True
    assert result.publication_observed is False
    assert result.exit_codes == (0, 0)
    for state in states(result):
        assert state["status"] == "refused"
        assert state["ddp_forward_calls"] == 0
        assert state["ddp_gradient_reductions"] == 0
        assert state["review_backward_calls"] == 0
        assert state["review_parameter_gradient_events"] == 0
        assert state["parameters_before"] == state["parameters_after"]


@pytest.mark.parametrize(
    ("kind", "phase", "delay"),
    ((FaultKind.RAISE_BEFORE_BACKWARD, FaultPhase.BEFORE_OPTIMIZER_STEP, 0.0),
     (FaultKind.SKIP_COLLECTIVE, FaultPhase.BEFORE_COLLECTIVE, 0.0),
     (FaultKind.STALL_BEFORE_COLLECTIVE, FaultPhase.BEFORE_COLLECTIVE, 6.0)),
)
def test_r2_reviewed_n_g4_fault_tests_preserve_evidence(tmp_path: Path, kind, phase, delay) -> None:
    result = run_native(
        tmp_path, fault_plan=FaultPlan(target_rank=1, kind=kind, phase=phase, delay_seconds=delay),
        bounds=FAULT_BOUNDS, extra_args=("--checkpoint-root", str(tmp_path / "dcp"),
                                         "--checkpoint-generation", "1"),
    )
    assert result.all_reaped is True
    assert result.watchdog_fired is False
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert all(Path(result.invocation_dir, f"state_{rank}.json").exists() for rank in (0, 1))
    assert all(Path(result.invocation_dir, f"rank_{rank}.log").exists() for rank in (0, 1))
    target_log = Path(result.invocation_dir, "rank_1.log").read_text()
    assert "ddp harness injected fault: rank 1" in target_log
    target_state = json.loads(Path(result.invocation_dir, "state_1.json").read_text())
    assert target_state["rank"] == 1, "target state artifact must identify culprit rank"
    assert "fault_kind" not in target_state or target_state["fault_kind"] == kind.name
    assert "fault_phase" not in target_state or target_state["fault_phase"] == phase.name
    assert not (tmp_path / "dcp" / "latest.json").exists()


@pytest.mark.parametrize(
    ("kind", "phase", "delay"),
    ((FaultKind.RAISE_BEFORE_BACKWARD, FaultPhase.BEFORE_OPTIMIZER_STEP, 0.0),
     (FaultKind.SKIP_COLLECTIVE, FaultPhase.BEFORE_COLLECTIVE, 0.0),
     (FaultKind.STALL_BEFORE_COLLECTIVE, FaultPhase.BEFORE_COLLECTIVE, 6.0)),
)
def test_r2_reviewer_n_g4_broad_exit_fact_oracle(tmp_path: Path, kind, phase, delay) -> None:
    result = run_native(
        tmp_path, fault_plan=FaultPlan(target_rank=1, kind=kind, phase=phase, delay_seconds=delay),
        bounds=FAULT_BOUNDS, extra_args=("--checkpoint-root", str(tmp_path / "dcp"),
                                         "--checkpoint-generation", "1"),
    )
    assert result.all_reaped is True
    assert all(code is not None for code in result.exit_codes)
    assert any(code != 0 for code in result.exit_codes), "broad exit fact: at least one nonzero child exit"


def test_r2_n_g3_resume_applies_detached_dcp_buffers(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    first = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "1",
                                             "--checkpoint-root", str(root), "--checkpoint-generation", "1"))
    assert_success(first)
    resumed = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "2",
                                                "--checkpoint-root", str(root), "--resume-generation", "1"))
    assert_success(resumed)
    before = states(resumed)
    saved = states(first)
    for current in before:
        previous = saved[current["rank"]]
        assert current["parameters_before"] == previous["parameters_after"]
        assert current["optimizer_state_before"] == previous["optimizer_state_after"]


def test_r2_n_g3_topology_gate_precedes_all_dcp_and_state_mutation(tmp_path: Path, monkeypatch) -> None:
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    runtime = Mock(rank=1, world_size=3)
    runtime.broadcast_object.return_value = {"world_size": 2, "path": "generations/gen-000001", "files": []}
    store = CheckpointPayloadStore(root=tmp_path, runtime=runtime)
    load = Mock(side_effect=AssertionError("DCP load entered"))
    apply = Mock(side_effect=AssertionError("state application entered"))
    runtime.all_gather_objects.return_value = (None, None)
    monkeypatch.setattr(checkpoint_module.dcp, "load", load)
    monkeypatch.setattr(checkpoint_module, "set_state_dict", apply)
    with pytest.raises(CheckpointTopologyMismatch):
        store.load(model, optimizer, generation=1)
    assert load.call_count == 0
    assert apply.call_count == 0
    assert all(torch.equal(model.state_dict()[key], value) for key, value in model_before.items())
    assert optimizer.state_dict() == optimizer_before


def test_r2_n_g3b_perturbed_rank_sidecar_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "perturbed"
    saved = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "1",
                                             "--checkpoint-root", str(root), "--checkpoint-generation", "1"))
    assert_success(saved)
    latest = json.loads((root / "latest.json").read_text())
    sidecar = root / latest["path"] / "sidecars" / "rank-00001.json"
    payload = json.loads(sidecar.read_text())
    payload["completed_updates"] += 1
    sidecar.write_text(json.dumps(payload, sort_keys=True))
    failed = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "2",
                                               "--checkpoint-root", str(root), "--resume-generation", "1"))
    assert failed.publication_observed is False
    assert all(state["status"] == "resume_failed" for state in states(failed))


def test_r2_n_g5_failed_publication_keeps_previous_generation_selectable(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    first = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "1",
                                             "--checkpoint-root", str(root), "--checkpoint-generation", "1"))
    assert_success(first)
    latest = json.loads((root / "latest.json").read_text())
    second = run_native(tmp_path, bounds=FAULT_BOUNDS, extra_args=("--experiment", "resume", "--iterations", "2",
                                                                     "--checkpoint-root", str(root), "--checkpoint-generation", "2",
                                                                     "--checkpoint-failure-rank", "1"))
    assert second.publication_observed is False
    assert all(state["status"] != "success" for state in states(second)), (
        "failed publication must retain checkpoint_pending state"
    )
    assert json.loads((root / "latest.json").read_text()) == latest


def test_r2_n_g5_digest_change_blocks_publication(tmp_path: Path, monkeypatch) -> None:
    runtime = Mock(rank=0, world_size=2)
    runtime.barrier.return_value = None
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = 0
    original_digest = checkpoint_module._digest

    def digest_then_change(path: Path, *, root: Path):
        nonlocal calls
        digest = original_digest(path, root=root)
        calls += 1
        if calls == 1:
            peer = root / "sidecars" / "rank-00001.json"
            peer.parent.mkdir(parents=True, exist_ok=True)
            peer.write_text("{}")
            path.write_text("{\"changed\": true}")
        return digest

    runtime.all_gather_objects.side_effect = lambda value: [value, value]
    monkeypatch.setattr(checkpoint_module, "_digest", digest_then_change)
    monkeypatch.setattr(checkpoint_module, "get_state_dict", lambda *_args: ({}, {}))
    monkeypatch.setattr(checkpoint_module.dcp, "save", lambda *_args, **_kwargs: None)
    with pytest.raises(checkpoint_module.CheckpointCorrupt, match="digest changed"):
        CheckpointPayloadStore(root=tmp_path, runtime=runtime).save(
            model, optimizer, generation=1, sampler_state={}, rng_state={}, completed_updates=1
        )


def test_r2_n_g5_delayed_writer_does_not_publish_partial_generation(tmp_path: Path) -> None:
    root = tmp_path / "delayed"
    first = run_native(tmp_path, extra_args=("--experiment", "resume", "--iterations", "1",
                                             "--checkpoint-root", str(root), "--checkpoint-generation", "1"))
    assert_success(first)
    latest = json.loads((root / "latest.json").read_text())
    second = run_native(tmp_path, bounds=FAULT_BOUNDS, extra_args=(
        "--experiment", "resume", "--iterations", "2", "--checkpoint-root", str(root),
        "--checkpoint-generation", "2", "--checkpoint-delay-rank", "1", "--checkpoint-delay-seconds", "6",
    ))
    assert second.publication_observed is False
    assert json.loads((root / "latest.json").read_text()) == latest


def test_r2_n_g6_reduction_count_does_not_scale_with_sampling_work(tmp_path: Path) -> None:
    short = run_native(tmp_path, extra_args=("--mcmc-steps", "1", "--kinetic-forwards", "1"))
    long = run_native(tmp_path, extra_args=("--mcmc-steps", "5", "--kinetic-forwards", "5"))
    assert_success(short)
    assert_success(long)
    for left, right in zip(states(short), states(long), strict=True):
        assert left["ddp_gradient_reductions_per_update"] == 1
        assert right["ddp_gradient_reductions_per_update"] == 1
        assert left["sampling_gradient_reductions"] == right["sampling_gradient_reductions"] == 0
        assert right["sampling_raw_model_calls"] > left["sampling_raw_model_calls"]


def test_r2_n_e2_coordinate_work_uses_raw_model_only(tmp_path: Path) -> None:
    result = run_native(tmp_path, extra_args=("--kinetic-forwards", "3"))
    assert_success(result)
    for state in states(result):
        assert state["ddp_forward_calls"] == 1, "coordinate work must not use DDP wrapper"
        assert state["access"]["coordinate_forward_owner"] == "raw_model", "coordinate owner"
        assert state["access"]["used_module_attribute"] is False


def test_r2_n_e3_optimizer_state_and_closure_objective_are_global(tmp_path: Path) -> None:
    result = run_native(tmp_path, extra_args=("--optimizer", "closure", "--closure-inner-iterates", "3"))
    assert_success(result)
    observed = states(result)
    assert observed[0]["parameters_after"] == observed[1]["parameters_after"], (
        "closure parameters must be globally synchronized"
    )
    assert observed[0]["optimizer_state_after"] == observed[1]["optimizer_state_after"], (
        "closure optimizer state must be globally synchronized"
    )
    for state in observed:
        assert state["synchronized_closure_calls"] == state["closure_calls"]
        assert state["final_gradient_call"] == state["closure_calls"]


def test_r2_n_e3_sgd_and_adam_have_nonempty_independent_optimizer_evidence(tmp_path: Path) -> None:
    for name in ("sgd", "adam"):
        result = run_native(tmp_path, extra_args=("--optimizer", name))
        assert_success(result)
        assert all(state["optimizer_state_after"]["state"] for state in states(result))
        if name == "sgd":
            for state in states(result):
                for key, values in state["parameters_before"].items():
                    expected = [
                        value - 0.05 * state["gradients"][key]["__tensor__"][0]
                        for value in values
                    ]
                    assert state["parameters_after"][key] == pytest.approx(expected, abs=1e-12)


def test_r2_n_e4_inventory_names_consumed_dcp_apis_and_classifies_them(tmp_path: Path) -> None:
    result = run_native(tmp_path)
    assert_success(result)
    inventory = states(result)[0]["api_inventory"]
    consumed = {"torch.distributed.checkpoint.state_dict.get_state_dict",
                "torch.distributed.checkpoint.state_dict.set_state_dict"}
    assert consumed <= set(inventory["experimental"])
    assert set(inventory["stable"]).isdisjoint(consumed)
    assert states(result)[0]["accelerator_execution"] is False


@pytest.mark.xfail(strict=True, reason="EXPECTED CONTRACT-RED: worker inventory omits consumed save/load")
def test_r2_n_e4_consumed_save_load_inventory_oracle_expected_contract_red(tmp_path: Path) -> None:
    result = run_native(tmp_path)
    assert_success(result)
    inventory = states(result)[0]["api_inventory"]
    buckets = {name: set(inventory.get(name, ())) for name in ("stable", "experimental", "spike_prototype")}
    for api in ("torch.distributed.checkpoint.save", "torch.distributed.checkpoint.load"):
        locations = [name for name, entries in buckets.items() if api in entries]
        assert len(locations) == 1, (
            f"CONTRACT-RED: consumed inventory must classify {api} in exactly one bucket; "
            f"observed={locations}"
        )


def test_r2_n_e5_state_and_receipt_are_rank_attributed(tmp_path: Path) -> None:
    result = run_native(tmp_path)
    assert_success(result)
    for receipt, state in zip(result.receipts, states(result), strict=True):
        assert state["rank"] == receipt.rank
        assert state["world_size"] == receipt.world_size
        assert state["pid"] == receipt.pid
        assert state["hostname"] == receipt.hostname
        assert state["phase_sequence"] == receipt.phase_sequence


def test_r2_checkpoint_load_uses_no_state_application_on_topology_mismatch(tmp_path: Path) -> None:
    """A second direct boundary check guards the pre-mutation contract."""
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    runtime = Mock(rank=0, world_size=2)
    runtime.broadcast_object.return_value = {"world_size": 3, "path": "generations/gen-000001", "files": []}
    generation = tmp_path / "generations" / "gen-000001"
    generation.mkdir(parents=True)
    (generation / "COMPLETE").write_text("COMPLETE\n")
    (generation / "manifest.json").write_text(json.dumps({"world_size": 3, "files": []}))
    with pytest.raises(CheckpointTopologyMismatch):
        CheckpointPayloadStore(root=tmp_path, runtime=runtime).load(model, optimizer, generation=1)
